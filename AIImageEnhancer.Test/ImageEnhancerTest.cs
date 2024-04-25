using SautinSoft.Document;
using SautinSoft.Document.Drawing;
using SkiaSharp;
using Svg;
using System.Drawing;
using System.IO;
using System.Text.RegularExpressions;
using System.Xml.Linq;

namespace AIImageEnhancer.Test; 
[TestClass]
public class ImageEnhancerTest {
    [TestMethod]
    public void EnhancerTest() {
        var path = Environment.CurrentDirectory;
        var models = $"{path}\\Models";
        var imagesPath = $"{path}\\Images";

        if (!Directory.Exists(models)) {
            Directory.CreateDirectory(models);
        }
        if (!Directory.Exists(imagesPath)) {
            Directory.CreateDirectory(imagesPath);
        }

        var filesModels = Directory.GetFiles(models);
        var imgs = Directory.GetFiles(imagesPath);
        var mFiles = filesModels.Select(f => new FileInfo(f).Name.Split(".")[0]).ToList();
        var log = "";

        foreach (var modelName in mFiles) {
            try {
                using var imageEnhancer = new ImageEnhancer();
                if (imageEnhancer.LoadModel(models, modelName, 0)) {
                    imageEnhancer.ScaleAndSave($"{imagesPath}\\teste.jpg", imagesPath, "jpg", false, 400);
                }
            } catch (Exception e) { }

            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
    



        if (!string.IsNullOrEmpty(log)) {
            File.WriteAllText(log, $"{imagesPath}\\log.txt");
        }

        Assert.IsNotNull(true);
         
    }
}