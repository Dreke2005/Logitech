using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text.Json;
using System.Threading;

namespace Loupedeck.PaperDeskPlugin;

public sealed class PaperDeskPlugin : Plugin
{
    public override bool UsesApplicationApiOnly => true;
    public override bool HasNoApplication => true;

    internal static readonly string DataPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "LogiInsightPaper");
    internal static event Action? KeyLabelsChanged;
    internal static volatile KeypadState? Displayed;
    private Timer? timer;

    public override void Load()
    {
        timer = new Timer(_ => Refresh(), null, TimeSpan.FromMilliseconds(350), TimeSpan.FromMilliseconds(350));
    }

    public override void Unload()
    {
        timer?.Dispose();
        Displayed = null;
    }

    private void Refresh()
    {
        var current = Read<KeypadState>("state.json");
        if (current?.labels?.Length != 9 ||
            Math.Abs(DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - current.updated) > 6000)
        {
            current = null;
        }
        var previous = Displayed;
        Displayed = current;
        if (previous?.session != current?.session || previous?.revision != current?.revision
            || previous?.cameraActive != current?.cameraActive
            || previous?.active != current?.active || previous?.status != current?.status
            || previous?.activeMenu != current?.activeMenu)
        {
            this.Log.Info(current is null ? "Paper workbench closed." : $"Paper menu revision {current.revision}.");
            KeyLabelsChanged?.Invoke();
        }
    }

    internal static T? Read<T>(string file)
    {
        try
        {
            using var stream = new FileStream(Path.Combine(DataPath, file), FileMode.Open, FileAccess.Read,
                FileShare.ReadWrite | FileShare.Delete);
            return JsonSerializer.Deserialize<T>(stream);
        }
        catch (IOException) { return default; }
        catch (UnauthorizedAccessException) { return default; }
        catch (JsonException) { return default; }
    }

    internal static void WriteCommand(object command)
    {
        var folder = Path.Combine(DataPath, "commands");
        Directory.CreateDirectory(folder);
        var file = Path.Combine(folder, $"{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds():D13}-{Guid.NewGuid():N}");
        File.WriteAllText(file + ".tmp", JsonSerializer.Serialize(command));
        File.Move(file + ".tmp", file + ".json");
    }

    internal sealed record KeypadState(string session, int revision, long updated, string[] labels,
        string? menu, bool cameraActive = false, string? active = null, string? status = null,
        string? activeMenu = null);
    internal sealed record Launcher(string python, string script);
}

public sealed class PaperDeskApplication : ClientApplication
{
    protected override string GetProcessName() => "";
    protected override string GetBundleName() => "";
    public override ClientApplicationStatus GetApplicationStatus() => ClientApplicationStatus.Unknown;
}

public sealed class PaperKey : PluginDynamicCommand
{
    private static long lastLaunch = -4000;

    // Keys 1-9 follow the workbench menu. Keys 10-18 and 19-27 always show version control and its
    // host actions, so keypad pages 2 and 3 reach them from any app.
    private const int Slots = 27;
    private static readonly string?[] PageMenus = { null, "git", "git_remote" };

    public PaperKey() : base()
    {
        for (int slot = 0; slot < Slots; ++slot)
        {
            string group = slot < 9 ? "Paper desk" : slot < 18 ? "Version control" : "Version control host";
            AddParameter(slot.ToString(CultureInfo.InvariantCulture), $"Paper key {slot + 1}", group);
        }
    }

    protected override bool OnLoad()
    {
        PaperDeskPlugin.KeyLabelsChanged += Redraw;
        return true;
    }

    protected override bool OnUnload()
    {
        PaperDeskPlugin.KeyLabelsChanged -= Redraw;
        return true;
    }

    private void Redraw()
    {
        for (int slot = 0; slot < Slots; ++slot)
        {
            ActionImageChanged(slot.ToString(CultureInfo.InvariantCulture));
        }
    }

    private static bool TrySlot(string parameter, out int slot) =>
        int.TryParse(parameter, NumberStyles.None, CultureInfo.InvariantCulture, out slot) && slot is >= 0 and < Slots;

    protected override string GetCommandDisplayName(string actionParameter, PluginImageSize imageSize) =>
        string.Empty;

    protected override BitmapImage GetCommandImage(string actionParameter, PluginImageSize imageSize)
    {
        if (!TrySlot(actionParameter, out var slot))
        {
            return PaperIcons.Read("blank", imageSize);
        }
        // Workbench closed: the start page, with the keypad's own keys and the webcam key that starts it.
        string menu = PageMenus[slot / 9] ?? PaperDeskPlugin.Displayed?.menu ?? "start";
        return PaperIcons.ForKey(menu, PaperAppFolder.ButtonForSlot(menu, slot % 9), imageSize);
    }

    protected override void RunCommand(string actionParameter)
    {
        if (!TrySlot(actionParameter, out int slot))
        {
            return;
        }
        var state = PaperDeskPlugin.Displayed;
        try
        {
            if (PageMenus[slot / 9] is { } fixedMenu)
            {
                if (state is null)
                {
                    OpenWorkbench("git");
                }
                else if (PaperAppFolder.ButtonForSlot(fixedMenu, slot % 9) is { } button)
                {
                    this.Plugin.Log.Info($"Paper key {fixedMenu}/{button[0]} pressed.");
                    PaperDeskPlugin.WriteCommand(new { state.session, menu = fixedMenu, action = button[0] });
                }
                return;
            }
            if ((state?.menu ?? "start") == "start")
            {
                // The start page: the stock keys work whether or not the workbench runs; the webcam key
                // starts the workbench (or its camera) and shows the apps.
                if (PaperAppFolder.ButtonForSlot("start", slot) is not { } button)
                {
                    return;
                }
                this.Plugin.Log.Info($"Start key {button[0]} pressed.");
                if (button[0] != "webcam")
                {
                    StockKeys.Run(button[0]);
                }
                else if (state is null)
                {
                    OpenWorkbench(webcam: true);
                }
                else
                {
                    PaperDeskPlugin.WriteCommand(new { state.session, menu = "start", action = "webcam" });
                }
                return;
            }
            if (state is null)
            {
                return;
            }
            if (state.labels[slot] == "")
            {
                return;
            }
            this.Plugin.Log.Info($"Paper key {slot + 1} pressed in {state.menu}.");
            PaperDeskPlugin.WriteCommand(new { state.session, state.revision, slot });
        }
        catch (IOException error) { this.Plugin.Log.Error(error.Message); }
        catch (UnauthorizedAccessException error) { this.Plugin.Log.Error(error.Message); }
        catch (System.ComponentModel.Win32Exception error) { this.Plugin.Log.Error(error.Message); }
    }

    internal static void OpenWorkbench(string? route = null, bool webcam = false)
    {
        long now = Environment.TickCount64;
        long previous = Interlocked.Read(ref lastLaunch);
        if (now - previous < 4000 || Interlocked.CompareExchange(ref lastLaunch, now, previous) != previous)
        {
            return;
        }
        var launcher = PaperDeskPlugin.Read<PaperDeskPlugin.Launcher>("launcher.json");
        if (launcher is null || !File.Exists(launcher.python) || !File.Exists(launcher.script))
        {
            return;
        }
        var start = new ProcessStartInfo(launcher.python) {
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = Path.GetDirectoryName(launcher.script)!,
        };
        start.ArgumentList.Add(launcher.script);
        start.ArgumentList.Add("--background");  // Started from the keypad: stay out of the way.
        if (webcam)
        {
            start.ArgumentList.Add("--webcam");
        }
        else if (route is not null)
        {
            start.ArgumentList.Add("--route");
            start.ArgumentList.Add(route);
        }
        using var process = Process.Start(start);
    }
}

/// <summary>The keypad's stock keys, kept on the start page: media, mute, emoji, screenshot, Explorer,
/// Logi Options+, and the Ask ChatGPT multi-action (copy, open ChatGPT, paste after a prompt).</summary>
internal static class StockKeys
{
    [System.Runtime.InteropServices.DllImport("user32.dll")]
    private static extern void keybd_event(byte key, byte scan, uint flags, UIntPtr extra);

    [System.Runtime.InteropServices.DllImport("user32.dll")]
    private static extern uint SendInput(uint count, Input[] inputs, int size);

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    private struct KeyboardInput
    {
        public ushort Key;
        public ushort Scan;
        public uint Flags;
        public uint Time;
        public IntPtr Extra;
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Explicit, Size = 40)]
    private struct Input
    {
        [System.Runtime.InteropServices.FieldOffset(0)] public uint Type;
        [System.Runtime.InteropServices.FieldOffset(8)] public KeyboardInput Keyboard;
    }

    private const uint KeyUp = 0x0002, Unicode = 0x0004;
    private const byte Win = 0x5B, Shift = 0x10, Control = 0x11, Enter = 0x0D;
    private const string ChatGptPrompt = "Please write a draft reply to the prompt below:";

    private static void Press(params byte[] keys)
    {
        foreach (var key in keys)
        {
            keybd_event(key, 0, 0, UIntPtr.Zero);
        }
        for (int index = keys.Length - 1; index >= 0; --index)
        {
            keybd_event(keys[index], 0, KeyUp, UIntPtr.Zero);
        }
    }

    private static void Type(string text)
    {
        var inputs = new Input[text.Length * 2];
        for (int index = 0; index < text.Length; ++index)
        {
            for (int part = 0; part < 2; ++part)
            {
                inputs[2 * index + part].Type = 1;
                inputs[2 * index + part].Keyboard.Scan = text[index];
                inputs[2 * index + part].Keyboard.Flags = Unicode | (part == 1 ? KeyUp : 0);
            }
        }
        SendInput((uint)inputs.Length, inputs, System.Runtime.InteropServices.Marshal.SizeOf<Input>());
    }

    private static void Launch(string file, string? folder = null)
    {
        using var process = Process.Start(new ProcessStartInfo(file) {
            UseShellExecute = true,
            WorkingDirectory = folder ?? "",
        });
    }

    internal static void Run(string key)
    {
        switch (key)
        {
            case "media_next": Press(0xB0); break;
            case "media_playpause": Press(0xB3); break;
            case "volume_mute": Press(0xAD); break;
            case "emoji": Press(Win, 0xBE); break;
            case "screenshot": Press(Win, Shift, 0x53); break;
            case "explorer": Launch("explorer.exe"); break;
            case "optionsplus":
                string folder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                    "LogiOptionsPlus");
                Launch(Path.Combine(folder, "logioptionsplus.exe"), folder);
                break;
            case "ask_chatgpt":
                ThreadPool.QueueUserWorkItem(_ =>
                {
                    Press(Control, 0x43);
                    Thread.Sleep(150);
                    Launch("https://chatgpt.com");
                    Thread.Sleep(1500);
                    Type(ChatGptPrompt);
                    Press(Shift, Enter);
                    Press(Control, 0x56);
                    Press(Enter);
                });
                break;
        }
    }
}
