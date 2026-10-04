// DAC Player for macOS: an app bundle that declares the com.descript.dac type (Info.plist) and
// handles Finder "open document" events: decode with a progress window, then open the WAV in the
// default audio app (Music / QuickTime Player / ...).
//   "DAC Player.app/Contents/MacOS/DAC Player" --register     make DAC Player the default for .dac
//   "DAC Player.app/Contents/MacOS/DAC Player" --unregister   re-register without claiming the default
#import <Cocoa/Cocoa.h>
#import <CoreServices/CoreServices.h>

#include "player.h"

#include <memory>
#include <string>
#include <thread>

using namespace dacn;

static NSString * NS(const std::string & s) { return [NSString stringWithUTF8String:s.c_str()]; }

static void error_alert(NSString * msg) {
    NSAlert * a = [[NSAlert alloc] init];
    a.alertStyle = NSAlertStyleCritical;
    a.messageText = @"DAC Player";
    a.informativeText = msg;
    [NSApp activateIgnoringOtherApps:YES];
    [a runModal];
}

static int do_register(bool claim_default) {
    NSURL * bundle = [[NSBundle mainBundle] bundleURL];
    OSStatus st = LSRegisterURL((__bridge CFURLRef) bundle, true);
    if (claim_default) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        st |= LSSetDefaultRoleHandlerForContentType(CFSTR("com.descript.dac"), kLSRolesAll,
                                                    (__bridge CFStringRef) [[NSBundle mainBundle] bundleIdentifier]);
#pragma clang diagnostic pop
    }
    printf("%s: %s\n", st == noErr ? "done" : "failed", claim_default ? ".dac files open with DAC Player" : "registration refreshed");
    return st == noErr ? 0 : 1;
}

@interface DacController : NSObject <NSApplicationDelegate>
@end

@implementation DacController {
    NSMutableArray<NSURL *> * _queue;
    NSWindow * _window;
    NSTextField * _label;
    NSProgressIndicator * _bar;
    NSTimer * _timer;
    std::unique_ptr<player_job> _job;
    std::unique_ptr<std::thread> _worker;
    fs::path _wav;
    BOOL _gotFiles;
}

- (instancetype)init {
    if ((self = [super init])) _queue = [NSMutableArray array];
    return self;
}

- (void)application:(NSApplication *)app openURLs:(NSArray<NSURL *> *)urls {
    _gotFiles = YES;
    [_queue addObjectsFromArray:urls];
    if (!_job) [self next];
}

- (void)applicationDidFinishLaunching:(NSNotification *)n {
    // launched without a document (e.g. from Finder or the Dock): explain and quit
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t) (0.7 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        if (self->_gotFiles) return;
        NSAlert * a = [[NSAlert alloc] init];
        a.messageText = @"DAC Player";
        a.informativeText = @"Double-click a .dac file in Finder to play it.";
        [NSApp activateIgnoringOtherApps:YES];
        [a runModal];
        [NSApp terminate:nil];
    });
}

- (void)buildWindow:(NSString *)title {
    _window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 440, 120)
                                          styleMask:NSWindowStyleMaskTitled
                                            backing:NSBackingStoreBuffered
                                              defer:NO];
    _window.title = @"DAC Player";
    NSView * v = _window.contentView;
    NSTextField * head = [NSTextField labelWithString:[@"♫ " stringByAppendingString:title]];
    head.font = [NSFont boldSystemFontOfSize:13];
    head.frame = NSMakeRect(20, 86, 400, 20);
    _label = [NSTextField labelWithString:@"Preparing..."];
    _label.frame = NSMakeRect(20, 62, 400, 18);
    _bar = [[NSProgressIndicator alloc] initWithFrame:NSMakeRect(20, 40, 400, 16)];
    _bar.indeterminate = NO;
    _bar.minValue = 0;
    _bar.maxValue = 1;
    NSButton * cancel = [NSButton buttonWithTitle:@"Cancel" target:self action:@selector(cancel:)];
    cancel.frame = NSMakeRect(340, 6, 80, 28);
    cancel.keyEquivalent = @"\033";
    for (NSView * s in @[head, _label, _bar, cancel]) [v addSubview:s];
    [_window center];
    [NSApp activateIgnoringOtherApps:YES];
    [_window makeKeyAndOrderFront:nil];
}

- (void)next {
    if (_queue.count == 0) { [NSApp terminate:nil]; return; }
    NSURL * url = _queue.firstObject;
    [_queue removeObjectAtIndex:0];
    const fs::path in = fs::u8path(url.fileSystemRepresentation);
    _wav = cached_wav_for(in);
    std::error_code ec;
    if (fs::exists(_wav, ec)) {
        fs::last_write_time(_wav, fs::file_time_type::clock::now(), ec);
        [self play];
        [self next];
        return;
    }
    _job = std::make_unique<player_job>(in, _wav);
    [self buildWindow:NS(_job->title())];
    player_job * job = _job.get();
    _worker = std::make_unique<std::thread>([job] { job->run(); });
    _timer = [NSTimer scheduledTimerWithTimeInterval:0.2 target:self selector:@selector(tick:) userInfo:nil repeats:YES];
}

- (void)tick:(NSTimer *)t {
    if (!_job->finished()) {
        _bar.doubleValue = _job->fraction();
        _label.stringValue = NS(_job->status());
        return;
    }
    [_timer invalidate];
    _worker->join();
    [_window orderOut:nil];
    if (_job->ok()) [self play];
    else if (!_job->cancelled()) error_alert([@"Could not decode this file:\n" stringByAppendingString:NS(_job->error_text())]);
    _job.reset();
    [self next];
}

- (void)cancel:(id)sender {
    if (_job) _job->cancel();
}

- (void)play {
    NSURL * url = [NSURL fileURLWithPath:NS(_wav.u8string())];
    if (![[NSWorkspace sharedWorkspace] openURL:url])
        error_alert([@"Could not open the decoded audio:\n" stringByAppendingString:url.path]);
}

@end

int main(int argc, const char ** argv) {
    @autoreleasepool {
        if (argc >= 2 && std::string(argv[1]) == "--register") return do_register(true);
        if (argc >= 2 && std::string(argv[1]) == "--unregister") return do_register(false);
        NSApplication * app = [NSApplication sharedApplication];
        DacController * c = [[DacController alloc] init];
        app.delegate = c;
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        [app run];
    }
    return 0;
}
