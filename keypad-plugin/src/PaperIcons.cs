using System;
using System.Collections.Generic;
using System.Linq;

namespace Loupedeck.PaperDeskPlugin;

internal static class PaperIcons
{
    private static readonly Dictionary<string, string> Aliases = new(StringComparer.OrdinalIgnoreCase)
    {
        ["architecture"] = "project",
        ["class_diagram"] = "project",
        ["default"] = "code",
        ["ide"] = "code",
        ["import"] = "paste",
        ["figure"] = "graph",
        ["languages"] = "language",
        ["matrix"] = "code",
        ["solve"] = "code",
        ["C++"] = "cpp",
    };

    private static readonly Dictionary<string, string> Resources = typeof(PaperIcons).Assembly
        .GetManifestResourceNames()
        .Where(name => name.EndsWith(".png", StringComparison.Ordinal))
        .ToDictionary(name => name.Split('.')[^2], StringComparer.OrdinalIgnoreCase);

    internal static string ForButton(string[]? button)
    {
        if (button is null)
        {
            return "blank";
        }
        string icon = button.Length > 2 ? button[2] : button[0];
        if (button[0] == "camera" && PaperDeskPlugin.Displayed?.cameraActive == true)
        {
            return "camera_stop";
        }
        return Aliases.TryGetValue(icon, out var alias) ? alias : icon;
    }

    /// <summary>Icon of a menu button, with a status bar when it is the key the workbench is working on.</summary>
    internal static BitmapImage ForKey(string? menu, string[]? button, PluginImageSize size)
    {
        var state = PaperDeskPlugin.Displayed;
        string? status = button is not null && state?.active == button[0]
            && (state.activeMenu is null || state.activeMenu == menu) ? state.status : null;
        return Read(ForButton(button), size, status, menu);
    }

    internal static BitmapImage Read(string icon, PluginImageSize size = PluginImageSize.None, string? status = null,
        string? menu = null)
    {
        // Each app page has its own colour set in icons/themed/<menu>__<icon>.png; other keys use the shared icon.
        string resource = menu is not null && Resources.TryGetValue($"{menu}__{icon}", out var themed) ? themed
            : Resources.TryGetValue(icon, out var found) ? found : Resources["blank"];
        var image = typeof(PaperIcons).Assembly.ReadImage(resource);
        int pixels = size switch
        {
            PluginImageSize.Width60 => 50,
            PluginImageSize.Width90 => 80,
            PluginImageSize.Width116 => 116,
            _ => 0,
        };
        if (pixels == 0 && status is null)
        {
            return image;
        }
        if (pixels == 0)
        {
            pixels = 116;
        }
        // Draw the symbol over the whole requested key so the host has no spare area for a title.
        using var builder = new BitmapBuilder(pixels, pixels);
        builder.Clear(new BitmapColor(24, 28, 36));
        builder.DrawImage(image, 0, 0, pixels, pixels);
        // Status of the pressed key: red while recording, amber while working, green when done, red frame on error.
        BitmapColor? color = status switch
        {
            "recording" => new BitmapColor(235, 64, 52),
            "busy" => new BitmapColor(245, 176, 40),
            "done" => new BitmapColor(52, 199, 110),
            "error" => new BitmapColor(235, 64, 52),
            _ => null,
        };
        if (color is { } bar)
        {
            int thickness = Math.Max(5, pixels / 11);
            builder.FillRectangle(0, pixels - thickness, pixels, thickness, bar);
            if (status == "error")
            {
                builder.FillRectangle(0, 0, pixels, thickness, bar);
            }
        }
        return builder.ToImage();
    }
}
