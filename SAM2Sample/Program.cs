using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SAM2Sharp;
using SkiaSharp;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;

public class Program
{
    public static async Task Main(string[] args) // async は SKBitmap.Decode が同期のため不要になる可能性
    {
        string encoderPath = "sam2_hiera_small.encoder.onnx"; // 実際のモデルパスに置き換えてください
        string decoderPath = "sam2_hiera_small.decoder.onnx"; // 実際のモデルパスに置き換えてください
        string imagePath = @"C:\Users\Binary_number\Downloads\部費\hq720.jpg"; // 実際の画像パス

        if (!File.Exists(encoderPath) || !File.Exists(decoderPath) || !File.Exists(imagePath))
        {
            Console.WriteLine("必要なファイルが見つかりません。パスを確認してください。");
            Console.WriteLine($"Encoder: {Path.GetFullPath(encoderPath)} (Exists: {File.Exists(encoderPath)})");
            Console.WriteLine($"Decoder: {Path.GetFullPath(decoderPath)} (Exists: {File.Exists(decoderPath)})");
            Console.WriteLine($"Image: {Path.GetFullPath(imagePath)} (Exists: {File.Exists(imagePath)})");
            return;
        }

        var generator = new SAM2Session(encoderPath, decoderPath); // pointsPerSide を調整

        // SkiaSharp の画像読み込み (同期)
        using (SKBitmap image = SKBitmap.Decode(imagePath))
        {
            if (image == null)
            {
                Console.WriteLine($"画像の読み込みに失敗しました: {imagePath}");
                return;
            }
            Console.WriteLine($"画像読み込み成功: {image.Width}x{image.Height}");

            List<SegmentationResult> masks = generator.GenerateMasks(image, pointsPerSide: 16);

            Console.WriteLine($"生成されたマスクの数 (フィルタ前): {masks.Count}");
            masks = ImageUtility.Deduplication(masks, 0.7f); // NMS/重複除去のしきい値を調整
            Console.WriteLine($"生成されたマスクの数 (フィルタ後): {masks.Count}");


            for (int i = 0; i < masks.Count; i++)
            {
                // PredictedIou のしきい値はモデルやタスクに応じて調整
                if (masks[i].PredictedIou > 0.80f && masks[i].Area > 100) // 面積でのフィルタも追加
                {
                    Console.WriteLine($"Mask {i}: IoU={masks[i].PredictedIou}, Area={masks[i].Area}, Box={masks[i].BoundingBox}");
                    // SaveMaskAsImage はモデル出力解像度のマスクを保存。
                    // 元画像にオーバーレイ表示する場合は、マスクのリサイズが必要。
                    ImageUtility.SaveMaskAsImage(masks[i].Mask, $"mask_{i}_iou{masks[i].PredictedIou:F2}.png");

                    // 元画像にバウンディングボックスとマスクを描画して保存する例
                    using SKBitmap originalImageWithMask = image.Copy(); // 元の画像をコピー
                    using SKCanvas canvas = new SKCanvas(originalImageWithMask);
                    using SKPaint boxPaint = new SKPaint
                    {
                        Color = SKColors.Red,
                        Style = SKPaintStyle.Stroke,
                        StrokeWidth = Math.Max(2, originalImageWithMask.Width / 200f) //線の太さを調整
                    };
                    canvas.DrawRect(masks[i].BoundingBox, boxPaint);

                    // マスクを元画像サイズにリサイズして描画 (オプション)
                    // この部分はより洗練されたリサイズと描画処理が必要
                    // SaveMaskAsImage で保存された bool[,] を SKBitmap に変換し、リサイズして描画
                    using SKBitmap maskBitmap = ImageUtility.CreateBitmapFromBoolMask(masks[i].Mask);
                    if (maskBitmap != null)
                    {
                        // BoundingBox の領域にマスクをアルファブレンドで描画
                        using SKBitmap resizedMask = maskBitmap.Resize(masks[i].BoundingBox.Size, new SKSamplingOptions(SKFilterMode.Nearest, SKMipmapMode.Nearest));
                        if (resizedMask != null)
                        {
                            using SKPaint maskPaint = new SKPaint
                            {
                                Color = new SKColor(0, 255, 0, 128), // 半透明の緑色
                                BlendMode = SKBlendMode.SrcOver
                            };
                            canvas.DrawBitmap(resizedMask, masks[i].BoundingBox.Location, maskPaint);
                            resizedMask.Dispose();
                        }
                        maskBitmap.Dispose();
                    }


                    using FileStream fs = new FileStream($"masked_image_{i}.png", FileMode.OpenOrCreate);
                    originalImageWithMask.Encode(SKEncodedImageFormat.Png, 90).SaveTo(fs);
                }
            }
        }
    }
}