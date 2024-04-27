using AIImageEnhancer.Helpers;
using Emgu.CV.Face;
using Helpers;
using ImageMagick;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIImageEnhancer;
public class ImageEnhancer : IDisposable {
    private InferenceSession _session;
    private string _modelName = "";

    public bool LoadModel(string modelPath, string modelName, int deviceId) {
        _modelName = modelName;
        var sessionOptions = new SessionOptions();
        if (deviceId == -1) {
            sessionOptions.AppendExecutionProvider_CPU();
        } else {
            sessionOptions.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
            sessionOptions.EnableMemoryPattern = false;
            sessionOptions.AppendExecutionProvider_DML(deviceId);
            sessionOptions.AppendExecutionProvider_CPU();
        }
        _session = new InferenceSession(Path.Combine(modelPath, $"{_modelName}.onnx"), sessionOptions);

        if (_session != null) {
            return true;
        }
        return false;
    }


    public void ScaleAndSave(List<string> inputPaths, string outputPath, string outputFormat, bool preserveAlpham, int overDimensionLimit = 300, int reductionPercentage = 0) {
        foreach (var inputPath in inputPaths) {
            ScaleAndSave(inputPath, outputPath, outputFormat, preserveAlpham, overDimensionLimit, reductionPercentage);
        }
    }

    public void ScaleAndSave(string inputPath, string outputPath, string outputFormat, bool preserveAlpha, int overDimensionLimit = 300, int reductionPercentage = 0) {
        var fileInfo = new FileInfo(inputPath);
        using Bitmap image = Scale(inputPath, preserveAlpha, overDimensionLimit, reductionPercentage);

        var saveName = $"{fileInfo.Name.Split(".")[0]}_{_modelName}.{outputFormat}";
        var savePath = $"{outputPath}\\{saveName}";

        Directory.CreateDirectory(outputPath);
        image?.Save(savePath);
    }

    public Bitmap? Scale(string inputPath, bool preserveAlpha = false, int overDimensionLimit = 300, int reductionPercentage = 0) {
        Bitmap image = null;

        var fileInfo = new FileInfo(inputPath);

        if (fileInfo.Extension is ".jpg" or ".jpeg" or "png") {
            image = new Bitmap(inputPath);
        } else {
            using var imageFromStream = new MagickImage(inputPath);
            using MemoryStream memStream = ToMemoryStream(imageFromStream);

            image = new Bitmap(memStream);
        }


        return Scale(image, preserveAlpha, overDimensionLimit, reductionPercentage);
    }

    public Bitmap? Scale(Bitmap? image, bool preserveAlpha, int overDimensionLimit, int reductionPercentage = 0) {
        if (image.Width > (overDimensionLimit * 2)) {
            image = Resize(image, (overDimensionLimit * 2));
        }

        if (image.Height > overDimensionLimit) {
            var partsEnchagend = new List<Bitmap>();
            var parts = ImageHelper.HorizontalSplit(image, overDimensionLimit);

            for (int i = 0; i < parts.Count; i++) {
                var part = parts[i];

                if (part.Width > overDimensionLimit) {
                    var partsBlockEnchagend = new List<Bitmap>();
                    var partsBlock = ImageHelper.VerticalSplit(part, overDimensionLimit);

                    for (int j = 0; j < partsBlock.Count; j++) {
                        var block = partsBlock[j];

                        var blockMelhoradaPart = Enhance(block, preserveAlpha, reductionPercentage);
                        partsBlockEnchagend.Add(blockMelhoradaPart);
                    }

                    var imgMelhoradaPart = ImageHelper.CombineOnSide(partsBlockEnchagend);
                    partsEnchagend.Add(imgMelhoradaPart);
                } else {
                    var imgMelhoradaPart = Enhance(part, preserveAlpha, reductionPercentage);
                    partsEnchagend.Add(imgMelhoradaPart);
                }

            }

            var saves = ImageHelper.CombineBelow(partsEnchagend);

            return saves;
        }

        return Enhance(image, preserveAlpha, reductionPercentage);
    }



    public Bitmap? Enhance(Bitmap? image, bool preserveAlpha, int reductionPercentage) {
        var originalPixelFormat = image.PixelFormat;
         
        Bitmap? alpha = null;
        if (preserveAlpha && originalPixelFormat != PixelFormat.Format24bppRgb) {
            if (image?.PixelFormat != PixelFormat.Format32bppArgb) {
                image = ImageProcess.ConvertBitmapToFormat(image, PixelFormat.Format32bppArgb);
            }
            ImageProcess.SplitChannel(image, out image, out alpha);
        }

        // Ensure that we got RGB channels in Format24bppRgb bitmap.
        if (image?.PixelFormat != PixelFormat.Format24bppRgb) {
            image = ImageProcess.ConvertBitmapToFormat(image, PixelFormat.Format24bppRgb);
        }

        var inMat = ConvertImageToFloatTensorUnsafe(image);

        var outMat = Inference(inMat);

        if (outMat == null) {
            image?.Dispose();
            return null;
        }

        image = ConvertFloatTensorToImageUnsafe(outMat);

        if (preserveAlpha && originalPixelFormat != PixelFormat.Format24bppRgb && alpha != null) {
            alpha = ImageProcess.ResizeAlphaChannel(alpha, image.Width, image.Height);    // Using BICUBIC to resize alpha channel.
            image = ImageProcess.CombineChannel(image, alpha);
        }

        if (reductionPercentage > 0 && reductionPercentage < 100) {
            image = ResizePerPercent(image, reductionPercentage);
        }

        return image;
    }

    public Tensor<float> Inference(Tensor<float> input) {
        var inputName = _session.InputMetadata.First().Key;
        var inputTensor = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor<float>(inputName, input) };
        var output = _session.Run(inputTensor).First().Value;
        return (Tensor<float>)output;
    }

    private MemoryStream Resize(MagickImage magickImage, int overLimit) {
        magickImage.Format = MagickFormat.Png;

        var Width = magickImage.Width;
        var Height = magickImage.Height;

        if (Width > overLimit) {
            var dif = Width - overLimit;

            if ((Height - dif) < (Height / 2)) {
                dif = Height - (Height / 2);
            }

            Width -= dif;
            Height -= dif;
        }

        MagickGeometry geometry = new(Width, Height);
        magickImage.Resize(geometry);
        return ToMemoryStream(magickImage);
    }

    private MemoryStream ResizePerPercent(MagickImage magickImage, int percent) {
        magickImage.Format = MagickFormat.Png;

        var Width = magickImage.Width;
        var Height = magickImage.Height;

        var reduncionWidth = (Width * percent) / 100;
        var reduncionHeight = (Height * percent) / 100;

        Width -= reduncionWidth;
        Height -= reduncionHeight;

        MagickGeometry geometry = new(Width, Height);
        magickImage.Resize(geometry);
        return ToMemoryStream(magickImage);
    }

    private Bitmap Resize(Bitmap? image, int overLimit) {
        var magikImage = BitmapToMagickImage(image);
        using var memStream = Resize(magikImage, overLimit);
        image = new Bitmap(memStream);
        return image;
    }

    private Bitmap ResizePerPercent(Bitmap? image, int percent) {
        var magikImage = BitmapToMagickImage(image);
        using var memStream = ResizePerPercent(magikImage, percent);
        image = new Bitmap(memStream);
        return image;
    }

    private MagickImage BitmapToMagickImage(Bitmap? image) {
        var m = new MagickFactory();
        using var ms = new MemoryStream();
        image.Save(ms, ImageFormat.Bmp);
        ms.Position = 0;
        MagickImage magikImage = new(m.Image.Create(ms));
        image.Dispose();

        return magikImage;
    }

    private MemoryStream ToMemoryStream(MagickImage magickImage) {
        var memStream = new MemoryStream();
        magickImage.Write(memStream);
        return memStream;
    }

    static void DarkenImage(Bitmap bmp, double multiplier) {
        for (int i = 0; i < bmp.Width; i++) {
            // Iterates over all the pixels
            for (int j = 0; j < bmp.Height; j++) {
                // Gets the current pixel
                var currentPixel = bmp.GetPixel(i, j);

                // Assigns each value the multiply, or the max value 255

                var newPixel = Color.FromArgb(
                    Math.Min((byte)255, (byte)(currentPixel.R * multiplier)),
                    Math.Min((byte)255, (byte)(currentPixel.G * multiplier)),
                    Math.Min((byte)255, (byte)(currentPixel.B * multiplier))
                    );

                // Sets the pixel 
                bmp.SetPixel(i, j, newPixel);
            }
        }
    }

    public Tensor<float> ConvertImageToFloatTensorUnsafe(Bitmap image) {
        // Create the Tensor with the appropiate dimensions for the NN
        Tensor<float> data = new DenseTensor<float>([1, 3, image.Width, image.Height]);

        BitmapData bmd = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, image.PixelFormat);
        int PixelSize = 3;

        unsafe {
            for (int y = 0; y < bmd.Height; y++) {
                // row is a pointer to a full row of data with each of its colors
                byte* row = (byte*)bmd.Scan0 + (y * bmd.Stride);
                for (int x = 0; x < bmd.Width; x++) {
                    // note the order of colors is BGR, convert to RGB
                    data[0, 0, x, y] = row[(x * PixelSize) + 2] / (float)255.0;
                    data[0, 1, x, y] = row[(x * PixelSize) + 1] / (float)255.0;
                    data[0, 2, x, y] = row[(x * PixelSize) + 0] / (float)255.0;
                }
            }

            image.UnlockBits(bmd);
        }
        return data;
    }

    public Bitmap ConvertFloatTensorToImageUnsafe(Tensor<float> tensor) {
        Bitmap bmp = new Bitmap(tensor.Dimensions[2], tensor.Dimensions[3], PixelFormat.Format24bppRgb);
        BitmapData bmd = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, bmp.PixelFormat);
        int PixelSize = 3;
        unsafe {
            for (int y = 0; y < bmd.Height; y++) {
                // row is a pointer to a full row of data with each of its colors
                byte* row = (byte*)bmd.Scan0 + (y * bmd.Stride);
                for (int x = 0; x < bmd.Width; x++) {
                    // note the order of colors is RGB, convert to BGR
                    // remember clamp to [0, 1]
                    row[x * PixelSize + 2] = (byte)(Math.Clamp(tensor[0, 0, x, y], 0, 1) * 255.0);
                    row[x * PixelSize + 1] = (byte)(Math.Clamp(tensor[0, 1, x, y], 0, 1) * 255.0);
                    row[x * PixelSize + 0] = (byte)(Math.Clamp(tensor[0, 2, x, y], 0, 1) * 255.0);
                }
            }

            bmp.UnlockBits(bmd);
        }
        return bmp;
    }

    public void Dispose() => _session?.Dispose();
}
