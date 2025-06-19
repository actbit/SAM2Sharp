using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Drawing.Imaging;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

public class SAM2Image
{
    private SAM2ImageEncoder encoder;
    private Tuple<int, int> origImSize;
    private string decoderPath;
    private Dictionary<int, SAM2ImageDecoder> decoders = new();
    private Dictionary<int, List<Tuple<int, int>>> pointCoords = new();
    private Dictionary<int, List<int>> pointLabels = new();
    private Dictionary<int, List<Tuple<int, int>>> boxCoords = new();
    private Dictionary<int, Bitmap> masks = new();
    private object[] imageEmbeddings;

    public SAM2Image(string encoderPath, string decoderPath)
    {
        encoder = new SAM2ImageEncoder(encoderPath);
        origImSize = encoder.InputShape;
        this.decoderPath = decoderPath;
    }

    public void SetImage(Bitmap image)
    {
        imageEmbeddings = encoder.Call(image);
        origImSize = new Tuple<int, int>(image.Height, image.Width);
        ResetPoints();
    }

    public Dictionary<int, Bitmap> AddPoint(Tuple<int, int> pt, bool isPositive, int labelId)
    {
        if (!decoders.ContainsKey(labelId))
            decoders[labelId] = new SAM2ImageDecoder(decoderPath, encoder.InputShape, origImSize);

        if (!pointCoords.ContainsKey(labelId))
        {
            pointCoords[labelId] = new List<Tuple<int, int>>() { pt };
            pointLabels[labelId] = new List<int>() { isPositive ? 1 : 0 };
        }
        else
        {
            pointCoords[labelId].Add(pt);
            pointLabels[labelId].Add(isPositive ? 1 : 0);
        }
        return DecodeMask(labelId);
    }

    public Dictionary<int, Bitmap> SetBox(Tuple<Tuple<int, int>, Tuple<int, int>> box, int labelId)
    {
        if (!decoders.ContainsKey(labelId))
            decoders[labelId] = new SAM2ImageDecoder(decoderPath, encoder.InputShape, origImSize);

        boxCoords[labelId] = new List<Tuple<int, int>>() { box.Item1, box.Item2 };
        return DecodeMask(labelId);
    }

    private Dictionary<int, Bitmap> DecodeMask(int labelId)
    {
        var (concatCoords, concatLabels) = MergePointsAndBoxes(labelId);
        var decoder = decoders[labelId];
        var highRes0 = (DenseTensor<float>)imageEmbeddings[0];
        var highRes1 = (DenseTensor<float>)imageEmbeddings[1];
        var imageEmbed = (DenseTensor<float>)imageEmbeddings[2];
        Bitmap mask;
        if (concatCoords.Count == 0)
        {
            mask = new Bitmap(origImSize.Item2, origImSize.Item1, PixelFormat.Format8bppIndexed);
        }
        else
        {
            (mask, _) = decoder.Call(imageEmbed, highRes0, highRes1, concatCoords, concatLabels);
        }
        masks[labelId] = mask;
        return masks;
    }

    private (List<Tuple<int, int>>, List<int>) MergePointsAndBoxes(int labelId)
    {
        var concatCoords = new List<Tuple<int, int>>();
        var concatLabels = new List<int>();
        bool hasPoints = pointCoords.ContainsKey(labelId);
        bool hasBoxes = boxCoords.ContainsKey(labelId);
        if (!hasPoints && !hasBoxes)
            return (new List<Tuple<int, int>>(), new List<int>());
        if (hasPoints)
        {
            concatCoords.AddRange(pointCoords[labelId]);
            concatLabels.AddRange(pointLabels[labelId]);
        }
        if (hasBoxes)
        {
            concatCoords.AddRange(boxCoords[labelId]);
            concatLabels.AddRange(new int[] { 2, 3 });
        }
        return (concatCoords, concatLabels);
    }

    public Dictionary<int, Bitmap> RemovePoint(Tuple<int, int> pt, int labelId)
    {
        if (!pointCoords.ContainsKey(labelId))
            return masks;

        var idx = pointCoords[labelId].FindIndex(x => x.Item1 == pt.Item1 && x.Item2 == pt.Item2);
        if (idx != -1)
        {
            pointCoords[labelId].RemoveAt(idx);
            pointLabels[labelId].RemoveAt(idx);
        }
        return DecodeMask(labelId);
    }

    public Dictionary<int, Bitmap> RemoveBox(int labelId)
    {
        if (boxCoords.ContainsKey(labelId))
            boxCoords.Remove(labelId);
        return DecodeMask(labelId);
    }

    public Dictionary<int, Bitmap> GetMasks()
    {
        return masks;
    }

    public void ResetPoints()
    {
        pointCoords.Clear();
        pointLabels.Clear();
        boxCoords.Clear();
        masks.Clear();
        decoders.Clear();
    }
}



public class SAM2ImageEncoder
{
    private InferenceSession session;
    public Tuple<int, int> InputShape { get; private set; }
    private string[] inputNames, outputNames;
    private int inputHeight, inputWidth;

    public SAM2ImageEncoder(string path)
    {
        session = new InferenceSession(path);
        var input = session.InputMetadata.First().Value;
        InputShape = new Tuple<int, int>(input.Dimensions[2], input.Dimensions[3]);
        inputNames = session.InputMetadata.Keys.ToArray();
        outputNames = session.OutputMetadata.Keys.ToArray();
        inputHeight = InputShape.Item1;
        inputWidth = InputShape.Item2;
    }

    public object[] Call(Bitmap image)
    {
        return EncodeImage(image);
    }

    private object[] EncodeImage(Bitmap image)
    {
        var inputTensor = PrepareInput(image);
        var outputs = Infer(inputTensor);
        return ProcessOutput(outputs);
    }

    private DenseTensor<float> PrepareInput(Bitmap image)
    {
        var resizedImage = new Bitmap(image, new Size(inputWidth, inputHeight));
        // 1.CV_BGR2RGB  
        for (int y = 0; y < resizedImage.Height; ++y)
            for (int x = 0; x < resizedImage.Width; ++x)
            {
                var c = resizedImage.GetPixel(x, y);
                resizedImage.SetPixel(x, y, Color.FromArgb(c.R, c.G, c.B)); // Assume input is already BGR  
            }
        // 2.Normalize (mean, std)  
        float[] mean = { 0.485f, 0.456f, 0.406f }, std = { 0.229f, 0.224f, 0.225f };
        var tensor = new DenseTensor<float>(new[] { 1, 3, inputHeight, inputWidth });
        for (int y = 0; y < inputHeight; ++y)
            for (int x = 0; x < inputWidth; ++x)
            {
                var c = resizedImage.GetPixel(x, y);
                tensor[0, 0, y, x] = ((c.R / 255f) - mean[0]) / std[0];
                tensor[0, 1, y, x] = ((c.G / 255f) - mean[1]) / std[1];
                tensor[0, 2, y, x] = ((c.B / 255f) - mean[2]) / std[2];
            }
        return tensor;
    }

    private IDisposableReadOnlyCollection<DisposableNamedOnnxValue> Infer(DenseTensor<float> inputTensor)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensor)
        };
        var watch = System.Diagnostics.Stopwatch.StartNew();
        var outputs = session.Run(inputs);
        watch.Stop();
        Console.WriteLine($"Encoder infer time: {watch.Elapsed.TotalMilliseconds:F2} ms");
        return outputs;
    }

    private object[] ProcessOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs)
    {
        return outputs.Select(o => o.Value).ToArray();
    }
}


public class SAM2ImageDecoder
{
    private InferenceSession session;
    private int scaleFactor = 4;
    private Tuple<int, int> origImSize, encoderInputSize;
    private string[] inputNames, outputNames;
    private float maskThreshold = 0.0f;
    public SAM2ImageDecoder(string path, Tuple<int, int> encoderInputSize, Tuple<int, int> origImSize = null, float maskThreshold = 0.0f)
    {
        session = new InferenceSession(path);
        this.encoderInputSize = encoderInputSize;
        this.origImSize = origImSize ?? encoderInputSize;
        this.maskThreshold = maskThreshold;
        inputNames = session.InputMetadata.Keys.ToArray();
        outputNames = session.OutputMetadata.Keys.ToArray();
    }

    public (Bitmap, float[]) Call(
        DenseTensor<float> imageEmbed, DenseTensor<float> highRes0, DenseTensor<float> highRes1,
        List<Tuple<int, int>> pointCoords, List<int> pointLabels)
    {
        var (inputPointCoords, inputPointLabels) = PreparePoints(pointCoords, pointLabels);
        int numLabels = inputPointLabels.Dimensions[1];
        var maskInput = new DenseTensor<float>(new[] { 1, 1, encoderInputSize.Item1 / scaleFactor, encoderInputSize.Item2 / scaleFactor });
        var hasMaskInput = new DenseTensor<float>(new[] { 1 });
        var originalSize = new DenseTensor<int>(new[] { 2 });
        originalSize[0] = origImSize.Item1;
        originalSize[1] = origImSize.Item2;

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputNames[0], imageEmbed),
            NamedOnnxValue.CreateFromTensor(inputNames[1], highRes0),
            NamedOnnxValue.CreateFromTensor(inputNames[2], highRes1),
            NamedOnnxValue.CreateFromTensor(inputNames[3], inputPointCoords),
            NamedOnnxValue.CreateFromTensor(inputNames[4], inputPointLabels),
            NamedOnnxValue.CreateFromTensor(inputNames[5], maskInput),
            NamedOnnxValue.CreateFromTensor(inputNames[6], hasMaskInput),
            //NamedOnnxValue.CreateFromTensor(inputNames[7], originalSize)
        };

        var watch = System.Diagnostics.Stopwatch.StartNew();
        var outputs = session.Run(inputs);
        watch.Stop();
        Console.WriteLine($"Decoder infer time: {watch.Elapsed.TotalMilliseconds:F2} ms");
        return ProcessOutput(outputs);
    }

    private (DenseTensor<float>, DenseTensor<float>) PreparePoints(List<Tuple<int, int>> pointCoords, List<int> pointLabels)
    {
        int N = pointCoords.Count;
        var coords = new DenseTensor<float>(new[] { 1, N, 2 });
        var labels = new DenseTensor<float>(new[] { 1, N });
        for (int i = 0; i < N; ++i)
        {
            float x = pointCoords[i].Item1 / (float)origImSize.Item2 * encoderInputSize.Item2;
            float y = pointCoords[i].Item2 / (float)origImSize.Item1 * encoderInputSize.Item1;
            coords[0, i, 0] = x;
            coords[0, i, 1] = y;
            labels[0, i] = pointLabels[i];
        }
        return (coords, labels);
    }

    private (Bitmap, float[]) ProcessOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs)
    {
        var scoresArr = ((DenseTensor<float>)outputs.ElementAt(1).Value).ToArray();
        var maskArr = ((DenseTensor<float>)outputs.ElementAt(0).Value).ToArray();
        // maskArr: [1,1,H,W] → flatten, H=height,W=width  
        int h = 256, w = 256;

        // 濃度0/255でbyte配列を作り直す  
        byte[] maskBytes = new byte[w * h];
        for (int i = 0; i < w * h; i++)
        {
            maskBytes[i] = maskArr[i] > maskThreshold ? (byte)255 : (byte)0;
        }

        // 8bitグレースケールBitmap生成  
        Bitmap maskBmp = new Bitmap(w, h, PixelFormat.Format8bppIndexed);
        ColorPalette pal = maskBmp.Palette;
        for (int i = 0; i < 256; ++i) pal.Entries[i] = Color.FromArgb(i, i, i);
        maskBmp.Palette = pal;

        BitmapData bmpData = maskBmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);

        int stride = bmpData.Stride;
        if (stride == w)
        {
            // そのままコピーできる場合  
            Marshal.Copy(maskBytes, 0, bmpData.Scan0, w * h);
        }
        else
        {
            // strideの都合で端数が生じる場合  
            byte[] tempRow = new byte[stride];
            for (int y = 0; y < h; y++)
            {
                Array.Clear(tempRow, 0, stride);
                Array.Copy(maskBytes, y * w, tempRow, 0, w);
                Marshal.Copy(tempRow, 0, bmpData.Scan0 + y * stride, stride);
            }
        }
        maskBmp.UnlockBits(bmpData);

        return (maskBmp, scoresArr);
    }
}




