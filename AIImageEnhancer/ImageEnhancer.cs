using AIImageEnhancer.Helpers;
using Helpers;
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


    public void Scale(List<string> inputPaths, string outputPath, string outputFormat, bool preserveAlpham, int overDimensionLimit = 300) {
        foreach (var inputPath in inputPaths) {
            ScaleAndSave(inputPath, outputPath, outputFormat, preserveAlpham, overDimensionLimit);
        }
    }

    public void ScaleAndSave(string inputPath, string outputPath, string outputFormat, bool preserveAlpha, int overDimensionLimit = 300) {
        var fileInfo = new FileInfo(inputPath);
        using Bitmap image = Scale(inputPath, preserveAlpha, overDimensionLimit);

        var saveName = $"{fileInfo.Name.Split(".")[0]}_{_modelName}.{outputFormat}";
        var savePath = $"{outputPath}\\{saveName}";

        Directory.CreateDirectory(outputPath);
        image?.Save(savePath);
    }

    public Bitmap? Scale(string inputPath, bool preserveAlpha = false, int overDimensionLimit = 300) {
        using var image = new Bitmap(inputPath);
        return Scale(preserveAlpha, image, overDimensionLimit);
    }

    public Bitmap? Scale(bool preserveAlpha, Bitmap? image, int overDimensionLimit = 300) {
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

                        var blockMelhoradaPart = Scale(preserveAlpha, block);
                        partsBlockEnchagend.Add(blockMelhoradaPart);
                    }

                    var imgMelhoradaPart = ImageHelper.CombineOnSide(partsBlockEnchagend);
                    partsEnchagend.Add(imgMelhoradaPart);
                } else {
                    var imgMelhoradaPart = Scale(preserveAlpha, part);
                    partsEnchagend.Add(imgMelhoradaPart);
                }

            }

            var saves = ImageHelper.CombineBelow(partsEnchagend);

            return saves;
        }

        return Scale(preserveAlpha, image);
    }

    public Bitmap? Scale(bool preserveAlpha, Bitmap? image) {
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

        return image;
    }

    public Tensor<float> Inference(Tensor<float> input) {
        var inputName = _session.InputMetadata.First().Key;
        var inputTensor = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor<float>(inputName, input) };
        var output = _session.Run(inputTensor).First().Value;
        return (Tensor<float>)output;
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

    public void Dispose() {
        _session?.Dispose();
    }
}
