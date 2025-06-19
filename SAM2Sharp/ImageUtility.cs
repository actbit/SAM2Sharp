using SkiaSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace SAM2Sharp
{
    public class ImageUtility
    {
        // 簡易的な重複除去 (IoU ベースの NMS の代わり)
        // より高度なNMSアルゴリズムを検討することも推奨
        public static List<SegmentationResult> Deduplication(List<SegmentationResult> masks, float overlapThreshold = 0.7f)
        {
            if (!masks.Any()) return masks;

            // PredictedIouが高い順、次に面積が大きい順でソート
            var sortedMasks = masks.OrderByDescending(m => m.PredictedIou)
                                   .ThenByDescending(m => m.Area)
                                   .ToList();

            var finalMasks = new List<SegmentationResult>();

            foreach (var currentMask in sortedMasks)
            {
                bool isDuplicate = false;
                foreach (var acceptedMask in finalMasks)
                {
                    // BoundingBox の IoU を計算 (より正確にはマスクピクセルの IoU)
                    float iou = CalculateRectIoU(currentMask.BoundingBox, acceptedMask.BoundingBox);
                    if (iou > overlapThreshold)
                    {
                        // マスクピクセルレベルでの重複も確認した方が良いが、ここでは BBox IoU で代用
                        // より厳密には、両方のマスクを同じ解像度にリサイズしてピクセル単位のIoUを計算する
                        isDuplicate = true;
                        break;
                    }
                }

                if (!isDuplicate)
                {
                    finalMasks.Add(currentMask);
                }
            }
            return finalMasks;
        }

        static float CalculateRectIoU(SKRectI rectA, SKRectI rectB)
        {
            int xA = Math.Max(rectA.Left, rectB.Left);
            int yA = Math.Max(rectA.Top, rectB.Top);
            int xB = Math.Min(rectA.Right, rectB.Right);
            int yB = Math.Min(rectA.Bottom, rectB.Bottom);

            int intersectionArea = Math.Max(0, xB - xA) * Math.Max(0, yB - yA);
            if (intersectionArea == 0) return 0f;

            int areaA = rectA.Width * rectA.Height;
            int areaB = rectB.Width * rectB.Height;

            float iou = intersectionArea / (float)(areaA + areaB - intersectionArea);
            return iou;
        }
        // bool[,] マスクからグレースケール SKBitmap を作成 (可視化やリサイズ用)
        public static SKBitmap CreateBitmapFromBoolMask(bool[,] maskData)
        {
            if (maskData == null) return null;
            int height = maskData.GetLength(0);
            int width = maskData.GetLength(1);
            if (width == 0 || height == 0) return null;

            var bitmap = new SKBitmap(width, height, SKColorType.Gray8, SKAlphaType.Opaque);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // Gray8 の場合、RGBは同じ値
                    byte grayValue = maskData[y, x] ? (byte)255 : (byte)0;
                    bitmap.SetPixel(x, y, new SKColor(grayValue, grayValue, grayValue));
                }
            }
            return bitmap;
        }
        public static void SaveMaskAsImage(bool[,] maskData, string outputPath)
        {
            if (maskData == null) return;
            int height = maskData.GetLength(0);
            int width = maskData.GetLength(1);
            if (width == 0 || height == 0) return;

            // SKColorType.Gray8 を使用してグレースケール画像を保存
            using (var bitmap = new SKBitmap(width, height, SKColorType.Gray8, SKAlphaType.Opaque))
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        // Gray8 の場合、RGBは同じ値 (輝度)
                        byte grayValue = maskData[y, x] ? (byte)255 : (byte)0;
                        // SetPixelはSKColorを取るので、(gray, gray, gray) で指定
                        bitmap.SetPixel(x, y, new SKColor(grayValue, grayValue, grayValue));
                    }
                }
                using (SKImage image = SKImage.FromBitmap(bitmap))
                using (SKData data = image.Encode(SKEncodedImageFormat.Png, 100)) // 100は品質 (PNGでは通常無視される)
                using (Stream stream = File.OpenWrite(outputPath))
                {
                    data.SaveTo(stream);
                }
            }
        }
    }
}
