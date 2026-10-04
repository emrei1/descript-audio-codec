// DAC Player for Linux / freedesktop desktops (GNOME, KDE, XFCE, ...): the .dac file handler.
//   dac-player <file.dac>      decode and open in the default media player (xdg-open)
//   dac-player --register      register the audio/x-dac MIME type and make DAC Player its default app
//   dac-player --unregister    undo --register
// Progress is shown with zenity when it is installed, otherwise with a desktop notification.
#include "player.h"

#include "paths.h"

#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <spawn.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char ** environ;

using namespace dacn;

namespace {

fs::path data_home() {
    const char * x = getenv("XDG_DATA_HOME");
    if (x && *x) return fs::path(x);
    const char * h = getenv("HOME");
    return fs::path(h ? h : ".") / ".local" / "share";
}

bool have(const char * prog) {
    const char * path = getenv("PATH");
    if (!path) return false;
    std::string p = path;
    size_t start = 0;
    while (start <= p.size()) {
        const size_t end = p.find(':', start);
        const std::string dir = p.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!dir.empty() && access((dir + "/" + prog).c_str(), X_OK) == 0) return true;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return false;
}

// Run a program and wait for it (no shell involved).
int run(std::vector<std::string> args) {
    std::vector<char *> av;
    for (auto & a : args) av.push_back(a.data());
    av.push_back(nullptr);
    pid_t pid;
    if (posix_spawnp(&pid, av[0], nullptr, nullptr, av.data(), environ) != 0) return -1;
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// Start a program without waiting for it.
void launch(std::vector<std::string> args) {
    std::vector<char *> av;
    for (auto & a : args) av.push_back(a.data());
    av.push_back(nullptr);
    pid_t pid;
    posix_spawnp(&pid, av[0], nullptr, nullptr, av.data(), environ);
}

void notify(const std::string & msg) {
    if (have("notify-send")) launch({"notify-send", "--app-name=DAC Player", "DAC Player", msg});
}

void show_error(const std::string & msg) {
    fprintf(stderr, "DAC Player: %s\n", msg.c_str());
    if (have("zenity")) run({"zenity", "--error", "--title=DAC Player", "--no-markup", "--text=" + msg});
    else if (have("kdialog")) run({"kdialog", "--title", "DAC Player", "--error", msg});
    else notify(msg);
}

std::string quote_exec(const std::string & s) {   // Desktop Entry Exec quoting
    std::string q = "\"";
    for (char c : s) { if (c == '"' || c == '`' || c == '$' || c == '\\') q += '\\'; q += c; }
    return q + "\"";
}

int do_register() {
    const fs::path exe = executable_path();
    const fs::path mime_dir = data_home() / "mime";
    const fs::path apps_dir = data_home() / "applications";
    std::error_code ec;
    fs::create_directories(mime_dir / "packages", ec);
    fs::create_directories(apps_dir, ec);

    std::ofstream(mime_dir / "packages" / "dac-player.xml")
        << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
           "<mime-info xmlns=\"http://www.freedesktop.org/standards/shared-mime-info\">\n"
           "  <mime-type type=\"audio/x-dac\">\n"
           "    <comment>Descript Audio Codec audio</comment>\n"
           "    <generic-icon name=\"audio-x-generic\"/>\n"
           "    <glob pattern=\"*.dac\"/>\n"
           "  </mime-type>\n"
           "</mime-info>\n";
    std::ofstream(apps_dir / "dac-player.desktop")
        << "[Desktop Entry]\n"
           "Type=Application\n"
           "Name=DAC Player\n"
           "Comment=Play Descript Audio Codec (.dac) files\n"
           "Exec=" << quote_exec(exe.string()) << " %f\n"
           "Icon=audio-x-generic\n"
           "Terminal=false\n"
           "NoDisplay=true\n"
           "MimeType=audio/x-dac;\n"
           "Categories=AudioVideo;Audio;Player;\n";

    bool ok = true;
    if (have("update-mime-database")) ok &= run({"update-mime-database", mime_dir.string()}) == 0;
    else fprintf(stderr, "warning: update-mime-database not found (install shared-mime-info)\n");
    if (have("update-desktop-database")) run({"update-desktop-database", apps_dir.string()});
    if (have("xdg-mime")) ok &= run({"xdg-mime", "default", "dac-player.desktop", "audio/x-dac"}) == 0;
    else fprintf(stderr, "warning: xdg-mime not found (install xdg-utils)\n");
    printf("%s: .dac files are associated with %s\n", ok ? "done" : "partially done", exe.c_str());
    return ok ? 0 : 1;
}

int do_unregister() {
    std::error_code ec;
    fs::remove(data_home() / "mime" / "packages" / "dac-player.xml", ec);
    fs::remove(data_home() / "applications" / "dac-player.desktop", ec);
    if (have("update-mime-database")) run({"update-mime-database", (data_home() / "mime").string()});
    if (have("update-desktop-database")) run({"update-desktop-database", (data_home() / "applications").string()});
    printf("done: the .dac association was removed\n");
    return 0;
}

// Decode with a zenity progress dialog; returns false if the user cancelled.
void run_with_progress(player_job & job) {
    std::thread worker([&job] { job.run(); });
    FILE * z = nullptr;
    if (have("zenity")) {
        const std::string cmd = "zenity --progress --title='DAC Player' --text='Preparing...' --percentage=0 --auto-close 2>/dev/null";
        z = popen(cmd.c_str(), "w");
    } else {
        notify("Decoding " + job.title() + "...");
    }
    while (!job.finished()) {
        usleep(200 * 1000);
        if (z) {
            const int pct = std::min(99, (int) (job.fraction() * 100));
            if (fprintf(z, "%d\n# %s\n", pct, (job.title() + " - " + job.status()).c_str()) < 0 || fflush(z) != 0) {
                job.cancel();   // zenity closed: the user pressed Cancel
                pclose(z);
                z = nullptr;
            }
        }
    }
    if (z) { fprintf(z, "100\n"); fflush(z); pclose(z); }
    worker.join();
}

int open_file(const fs::path & in) {
    std::error_code ec;
    if (!fs::exists(in, ec)) { show_error("File not found: " + in.string()); return 1; }
    const fs::path wav = cached_wav_for(in);
    if (!fs::exists(wav, ec)) {
        player_job job(in, wav);
        run_with_progress(job);
        if (!job.ok()) {
            if (!job.cancelled()) show_error("Could not decode " + in.filename().string() + ": " + job.error_text());
            return 1;
        }
    } else {
        fs::last_write_time(wav, fs::file_time_type::clock::now(), ec);
    }
    if (!have("xdg-open")) { show_error("xdg-open was not found; the decoded audio is at " + wav.string()); return 1; }
    launch({"xdg-open", wav.string()});
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    signal(SIGPIPE, SIG_IGN);   // a closed zenity pipe must not kill us
    if (argc >= 2 && std::string(argv[1]) == "--register") return do_register();
    if (argc >= 2 && std::string(argv[1]) == "--unregister") return do_unregister();
    if (argc >= 2) return open_file(fs::path(argv[1]));
    printf("DAC Player - plays Descript Audio Codec (.dac) files\n"
           "  dac-player <file.dac>     decode and open in the default media player\n"
           "  dac-player --register     associate .dac files with DAC Player\n"
           "  dac-player --unregister   remove the association\n");
    return 0;
}
