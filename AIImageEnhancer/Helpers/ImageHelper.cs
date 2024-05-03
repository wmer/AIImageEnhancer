using ImageMagick;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Text;

namespace Helpers;
public class ImageHelper {
    public static List<Bitmap> HorizontalSplit(Bitmap src, int pixels) {
        var bmps = new List<Bitmap>();

        for (int i = 0; i < src.Height; i += pixels) {
            var dst = new Bitmap(src.Width, pixels);
            using var grD = Graphics.FromImage(dst);
            grD.DrawImage(src, new Rectangle(0, 0, src.Width, pixels), new Rectangle(0, i, src.Width, pixels), GraphicsUnit.Pixel);
            bmps.Add(CropImage(dst));
        }

        return bmps;
    }

    public static List<Bitmap> VerticalSplit(Bitmap src, int pixels) {
        var bmps = new List<Bitmap>();

        for (int i = 0; i < src.Width; i += pixels) {
            var dst = new Bitmap(pixels, src.Height);
            using var grD = Graphics.FromImage(dst);
            grD.DrawImage(src, new Rectangle(0, 0, pixels, src.Height), new Rectangle(i, 0, pixels, src.Height), GraphicsUnit.Pixel);
            bmps.Add(CropImage(dst));
        }

        return bmps;
    }

    public static Bitmap CombineBelow(List<Bitmap> sources) {
        List<int> imageHeights = [];
        List<int> imageWidths = [];
        for (int i = 0; i < sources.Count; i++) {
            Bitmap img = sources[i];
            imageHeights.Add(img.Height);
            imageWidths.Add(img.Width);
        }

        int stitchedHeight = 0;
        Bitmap result = new Bitmap(imageWidths.Max(), imageHeights.Sum());
        using (Graphics g = Graphics.FromImage(result)) {
            for (int i = 0; i < sources.Count; i++) {
                Bitmap img = sources[i];
                g.DrawImage(img, 0, stitchedHeight);
                stitchedHeight += img.Height;
            }

        }
        return result;
    }

    public static Bitmap CombineOnSide(List<Bitmap> sources) {
        List<int> imageHeights = [];
        List<int> imageWidths = [];
        for (int i = 0; i < sources.Count; i++) {
            Bitmap img = sources[i];
            imageHeights.Add(img.Height);
            imageWidths.Add(img.Width);
        }

        int stitchedWidth = 0;
        Bitmap result = new Bitmap(imageWidths.Sum(), imageHeights.Max());
        using (Graphics g = Graphics.FromImage(result)) {
            for (int i = 0; i < sources.Count; i++) {
                using Bitmap img = sources[i];
                g.DrawImage(img, stitchedWidth, 0);
                stitchedWidth += img.Width;
            }
        }
        return result;

    }

    public static Bitmap Resize(Bitmap? image, int overLimit) {
        var magikImage = BitmapToMagickImage(image);
        var nImage = Resize(magikImage, overLimit);
        return nImage;
    }

    public static Bitmap ResizePerPercent(Bitmap? image, int percent) {
        var magikImage = BitmapToMagickImage(image);
        image = ResizePerPercent(magikImage, percent);
        return image;
    }

    public static Bitmap Resize(MagickImage magickImage, int overLimit) {
        var Width = magickImage.Width;
        var Height = magickImage.Height;
        var percentHeigth = (Height * 60) / 100;

        if (Width > overLimit) {
            var dif = Width - overLimit;

            if ((Height - dif) < percentHeigth) {
                dif = Height - percentHeigth;
            }

            Width -= dif;
            Height -= dif;
        }

        MagickGeometry geometry = new(Width, Height);
        magickImage.Resize(geometry);
        var image = MagickImageToBitmap(magickImage);
        return image;
    }

    public static Bitmap ResizePerPercent(MagickImage magickImage, int percent) {
        var Width = magickImage.Width;
        var Height = magickImage.Height;

        var reduncionWidth = (Width * percent) / 100;
        var reduncionHeight = (Height * percent) / 100;

        Width -= reduncionWidth;
        Height -= reduncionHeight;

        MagickGeometry geometry = new(Width, Height);
        magickImage.Resize(geometry);
        var image = MagickImageToBitmap(magickImage);
        return image;
    }

    public static Bitmap IncraseBrightnessContrast(Bitmap? image, int brilho, int contrast) {
        var magikImage = BitmapToMagickImage(image);
        magikImage.BrightnessContrast(new Percentage(brilho), new Percentage(contrast));
        var nImage = MagickImageToBitmap(magikImage);
        return nImage;
    }

    public static MagickImage BitmapToMagickImage(Bitmap? image) {
        var m = new MagickFactory();
        using var ms = new MemoryStream();
        image.Save(ms, ImageFormat.Bmp);
        ms.Position = 0;
        MagickImage magikImage = new(m.Image.Create(ms)) {
            Format = MagickFormat.Png
        };

        image.Dispose();

        return magikImage;
    }

    public static Bitmap MagickImageToBitmap(MagickImage magickImage) {
        using var memStream = new MemoryStream();
        magickImage.Write(memStream);
        var image = new Bitmap(memStream);
        magickImage.Dispose();
        return image;
    }

    //public static Bitmap CropImage(Bitmap image) {
    //    var mahickImage = BitmapToMagickImage(image);

    //    int height = mahickImage.Page.Height;
    //    int width = mahickImage.Page.Width;

    //    mahickImage.ColorFuzz = new Percentage(0);

    //    mahickImage.Trim();

    //    var nImage = MagickImageToBitmap(mahickImage);
    //    return nImage;
    //}

    public static Bitmap CropImage(Bitmap image) {
        // Get the non-transparent bounds of the image
        Rectangle bounds = GetNonTransparentBounds(image);

        // Create a new bitmap with the cropped size
        Bitmap croppedImage = new Bitmap(bounds.Width, bounds.Height);

        // Copy the cropped image data to the new bitmap
        using (Graphics g = Graphics.FromImage(croppedImage)) {
            g.DrawImage(image, new Rectangle(0, 0, croppedImage.Width, croppedImage.Height),
                        bounds, GraphicsUnit.Pixel);
        }

        return croppedImage;
    }

    public static Rectangle GetNonTransparentBounds(Bitmap image) {
        // Initialize bounds to the entire image
        Rectangle bounds = new Rectangle(0, 0, image.Width, image.Height);

        // Find the top edge of the image
        for (int y = 0; y < image.Height; y++) {
            for (int x = 0; x < image.Width; x++) {
                if (image.GetPixel(x, y).A != 0) {
                    bounds.Y = y;
                    goto bottom;
                }
            }
        }
    bottom:

        // Find the bottom edge of the image
        for (int y = image.Height - 1; y >= 0; y--) {
            for (int x = 0; x < image.Width; x++) {
                if (image.GetPixel(x, y).A != 0) {
                    bounds.Height = y - bounds.Y + 1;
                    goto right;
                }
            }
        }
    right:

        // Find the right edge of the image
        for (int x = image.Width - 1; x >= 0; x--) {
            for (int y = bounds.Y; y < bounds.Y + bounds.Height; y++) {
                if (image.GetPixel(x, y).A != 0) {
                    bounds.Width = x - bounds.X + 1;
                    goto left;
                }
            }
        }
    left:

        // Find the left edge of the image
        for (int x = 0; x < image.Width; x++) {
            for (int y = bounds.Y; y < bounds.Y + bounds.Height; y++) {
                if (image.GetPixel(x, y).A != 0) {
                    bounds.X = x;
                    goto done;
                }
            }
        }
    done:

        return bounds;
    }
}
