// DAC Player for Windows: the .dac file handler.
//   dac-player.exe <file.dac>      decode (progress dialog) and open in the default media player
//   dac-player.exe --register      associate .dac with this program for the current user (HKCU)
//   dac-player.exe --unregister    remove the association
#include "player.h"
#include "paths.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>
#include <shlobj.h>

#include <thread>

#pragma comment(linker, "\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

using namespace dacn;

namespace {

const wchar_t * kTitle = L"DAC Player";
const wchar_t * kProgId = L"DacPlayer.dac";

std::wstring W(const std::string & s) { return fs::u8path(s).wstring(); }

void error_box(const std::wstring & msg) { MessageBoxW(nullptr, msg.c_str(), kTitle, MB_ICONERROR | MB_OK); }

bool set_value(const std::wstring & sub, const wchar_t * name, const std::wstring & val) {
    HKEY k;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, sub.c_str(), 0, nullptr, 0, KEY_SET_VALUE, nullptr, &k, nullptr) != ERROR_SUCCESS) return false;
    const LONG r = RegSetValueExW(k, name, 0, REG_SZ, (const BYTE *) val.c_str(), (DWORD) ((val.size() + 1) * sizeof(wchar_t)));
    RegCloseKey(k);
    return r == ERROR_SUCCESS;
}

int do_register(bool quiet) {
    const std::wstring exe = executable_path().wstring();
    const std::wstring cls = L"Software\\Classes\\";
    const std::wstring cmd = L"\"" + exe + L"\" \"%1\"";
    bool ok = true;
    ok &= set_value(cls + L".dac", nullptr, kProgId);
    ok &= set_value(cls + L".dac", L"Content Type", W(kMimeType));
    ok &= set_value(cls + L".dac", L"PerceivedType", L"audio");
    ok &= set_value(cls + L".dac\\OpenWithProgids", kProgId, L"");
    ok &= set_value(cls + kProgId, nullptr, L"Descript Audio Codec audio");
    ok &= set_value(cls + kProgId + L"\\DefaultIcon", nullptr, L"\"" + exe + L"\",0");
    ok &= set_value(cls + kProgId + L"\\shell", nullptr, L"open");
    ok &= set_value(cls + kProgId + L"\\shell\\open", nullptr, L"Play");
    ok &= set_value(cls + kProgId + L"\\shell\\open\\command", nullptr, cmd);
    ok &= set_value(cls + L"Applications\\dac-player.exe\\SupportedTypes", L".dac", L"");
    ok &= set_value(cls + L"Applications\\dac-player.exe\\shell\\open\\command", nullptr, cmd);
    SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
    if (!quiet) {
        if (ok) MessageBoxW(nullptr, L".dac files will now open with DAC Player.", kTitle, MB_ICONINFORMATION | MB_OK);
        else error_box(L"Could not write the file association to the registry.");
    }
    return ok ? 0 : 1;
}

int do_unregister(bool quiet) {
    const std::wstring cls = L"Software\\Classes\\";
    RegDeleteTreeW(HKEY_CURRENT_USER, (cls + kProgId).c_str());
    RegDeleteTreeW(HKEY_CURRENT_USER, (cls + L"Applications\\dac-player.exe").c_str());
    HKEY k;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, (cls + L".dac").c_str(), 0, KEY_READ, &k) == ERROR_SUCCESS) {
        wchar_t v[128] = {};
        DWORD sz = sizeof v;
        const bool ours = RegQueryValueExW(k, nullptr, nullptr, nullptr, (BYTE *) v, &sz) == ERROR_SUCCESS && std::wstring(v) == kProgId;
        RegCloseKey(k);
        if (ours) RegDeleteTreeW(HKEY_CURRENT_USER, (cls + L".dac").c_str());
        else RegDeleteKeyValueW(HKEY_CURRENT_USER, (cls + L".dac\\OpenWithProgids").c_str(), kProgId);
    }
    SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
    if (!quiet) MessageBoxW(nullptr, L"The .dac file association was removed.", kTitle, MB_ICONINFORMATION | MB_OK);
    return 0;
}

bool play(const fs::path & wav) {
    const HINSTANCE r = ShellExecuteW(nullptr, L"open", wav.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    if ((INT_PTR) r <= 32) {
        error_box(L"Could not open the decoded audio (is there a default media player for .wav files?):\n" + wav.wstring());
        return false;
    }
    return true;
}

struct dialog_state {
    player_job * job;
    HWND hwnd = nullptr;
};

HRESULT CALLBACK dlg_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM, LONG_PTR ref) {
    auto * st = (dialog_state *) ref;
    switch (msg) {
        case TDN_CREATED:
            st->hwnd = hwnd;
            SendMessageW(hwnd, TDM_SET_PROGRESS_BAR_RANGE, 0, MAKELPARAM(0, 1000));
            break;
        case TDN_TIMER:
            if (st->job->finished()) { PostMessageW(hwnd, TDM_CLICK_BUTTON, IDCANCEL, 0); break; }
            SendMessageW(hwnd, TDM_SET_PROGRESS_BAR_POS, (WPARAM) (st->job->fraction() * 1000), 0);
            {
                const std::wstring s = W(st->job->status());
                SendMessageW(hwnd, TDM_SET_ELEMENT_TEXT, TDE_CONTENT, (LPARAM) s.c_str());
            }
            break;
        case TDN_BUTTON_CLICKED:
            if (wp == IDCANCEL && !st->job->finished()) st->job->cancel();
            return S_OK;
    }
    return S_OK;
}

int open_file(const fs::path & in) {
    std::error_code ec;
    if (!fs::exists(in, ec)) { error_box(L"File not found:\n" + in.wstring()); return 1; }
    const fs::path wav = cached_wav_for(in);
    if (fs::exists(wav, ec)) {
        fs::last_write_time(wav, fs::file_time_type::clock::now(), ec);   // keep it in the cache
        return play(wav) ? 0 : 1;
    }

    player_job job(in, wav);
    std::thread worker([&job] { job.run(); });
    dialog_state st{&job};
    const std::wstring main = L"♫ " + W(job.title());
    TASKDIALOGCONFIG cfg = {sizeof cfg};
    cfg.dwFlags = TDF_SHOW_PROGRESS_BAR | TDF_CALLBACK_TIMER | TDF_ALLOW_DIALOG_CANCELLATION | TDF_SIZE_TO_CONTENT;
    cfg.dwCommonButtons = TDCBF_CANCEL_BUTTON;
    cfg.pszWindowTitle = kTitle;
    cfg.pszMainInstruction = main.c_str();
    cfg.pszContent = L"Preparing...";
    cfg.pfCallback = dlg_proc;
    cfg.lpCallbackData = (LONG_PTR) &st;
    TaskDialogIndirect(&cfg, nullptr, nullptr, nullptr);
    if (!job.finished()) job.cancel();   // dialog closed early
    worker.join();
    if (!job.ok()) {
        if (!job.cancelled()) error_box(L"Could not decode this file:\n" + W(job.error_text()));
        return 1;
    }
    return play(wav) ? 0 : 1;
}

}  // namespace

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    int argc = 0;
    wchar_t ** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    bool quiet = false;
    for (int i = 1; i < argc; ++i) if (std::wstring(argv[i]) == L"--quiet") quiet = true;
    int rc = 0;
    if (argc >= 2 && std::wstring(argv[1]) == L"--register") rc = do_register(quiet);
    else if (argc >= 2 && std::wstring(argv[1]) == L"--unregister") rc = do_unregister(quiet);
    else if (argc >= 2) rc = open_file(fs::path(argv[1]));
    else
        MessageBoxW(nullptr,
                    L"Double-click a .dac file to play it.\n\n"
                    L"dac-player.exe --register     open .dac files with DAC Player\n"
                    L"dac-player.exe --unregister   remove the association",
                    kTitle, MB_ICONINFORMATION | MB_OK);
    LocalFree(argv);
    return rc;
}
