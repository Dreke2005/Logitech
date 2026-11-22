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

    internal static BitmapImage Read(string icon, PluginImageSize size = PluginImageSize.None)
    {
        string resource = Resources.TryGetValue(icon, out var found) ? found : Resources["blank"];
        var image = typeof(PaperIcons).Assembly.ReadImage(resource);
        int pixels = size switch
        {
            PluginImageSize.Width60 => 50,
            PluginImageSize.Width90 => 80,
            PluginImageSize.Width116 => 116,
            _ => 0,
        };
        if (pixels == 0)
        {
            return image;
        }
        // Draw the symbol over the whole requested key so the host has no spare area for a title.
        using var builder = new BitmapBuilder(pixels, pixels);
        builder.Clear(new BitmapColor(24, 28, 36));
        builder.DrawImage(image, 0, 0, pixels, pixels);
        return builder.ToImage();
    }
}
