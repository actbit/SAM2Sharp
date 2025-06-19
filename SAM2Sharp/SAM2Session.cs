using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SAM2Sharp;
using SkiaSharp; // SixLabors.ImageSharp を SkiaSharp に変更
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Threading.Tasks;
using System.Xml.Linq;
using static System.Net.Mime.MediaTypeNames;
namespace SAM2Sharp
{
    public class SAM2Session
    {
        private InferenceSession _imageEncoderSession;
        private InferenceSession _maskDecoderSession;

        //private readonly int _pointsPerSide;
        //private readonly float _predIouThresh;
        //private readonly int _minMaskRegionArea;

        private const string EncoderInputName = "image";
        private const string EncoderOutputName = "image_embed";
        private const string DecoderOutputNameHighResFeats_0 = "hidden_states_0"; // 例: sam-vit-base の場合など (要確認)
        private const string DecoderOutputNameHighResFeats_1 = "hidden_states_1"; // 例: (要確認)

        private const string DecoderEmbeddingInputName = "image_embed"; // モデルの入力名に合わせる
        private const string DecoderPointCoordsInputName = "point_coords";
        private const string DecoderPointLabelsInputName = "point_labels";
        // private const string DecoderOrigImSizeInputName = "orig_im_size"; // SAM 2 では使われない可能性あり
        private const string DecoderOutputMasksName = "masks"; // モデルの出力名に合わせる
        private const string DecoderOutputIouScoresName = "iou_predictions"; // モデルの出力名に合わせる


        public SAM2Session(string encoderModelPath, string decoderModelPath)
        {
            _imageEncoderSession = new InferenceSession(encoderModelPath);
            _maskDecoderSession = new InferenceSession(decoderModelPath);



            // ONNXモデルの入力/出力名を確認 (オプション)
            // Console.WriteLine("Encoder Inputs: " + string.Join(", ", _imageEncoderSession.InputMetadata.Keys));
            // Console.WriteLine("Encoder Outputs: " + string.Join(", ", _imageEncoderSession.OutputMetadata.Keys));
            // Console.WriteLine("Decoder Inputs: " + string.Join(", ", _maskDecoderSession.InputMetadata.Keys));
            // Console.WriteLine("Decoder Outputs: " + string.Join(", ", _maskDecoderSession.OutputMetadata.Keys));
        }
        public SAM2Session(byte[] encoderBytes, byte[] decoderBytes)
        {
            _imageEncoderSession = new InferenceSession(encoderBytes);
            _maskDecoderSession = new InferenceSession(decoderBytes);


            // ONNXモデルの入力/出力名を確認 (オプション)
            // Console.WriteLine("Encoder Inputs: " + string.Join(", ", _imageEncoderSession.InputMetadata.Keys));
            // Console.WriteLine("Encoder Outputs: " + string.Join(", ", _imageEncoderSession.OutputMetadata.Keys));
            // Console.WriteLine("Decoder Inputs: " + string.Join(", ", _maskDecoderSession.InputMetadata.Keys));
            // Console.WriteLine("Decoder Outputs: " + string.Join(", ", _maskDecoderSession.OutputMetadata.Keys));
        }
        public List<SegmentationResult> GenerateMasks(SKBitmap image, List<SKPoint> points, float predIouThresh = 0.88f, int minMaskRegionArea = 0) // Image<Rgb24> を SKBitmap に変更
        {
            var originalWidth = image.Width;
            var originalHeight = image.Height;

            var inputTensor = PreprocessImage(image, 1024);

            var encoderInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(EncoderInputName, inputTensor)
        };

            Tensor<float> imageEmbeddings;
            Tensor<float> highResFeats0Value = null; // 初期化
            Tensor<float> highResFeats1Value = null; // 初期化

            using var encoderResults = _imageEncoderSession.Run(encoderInputs);
            // SAM 2 のエンコーダ出力は通常 'image_embeds' と 'intermediate_hidden_states' (リスト)
            // ここでは 'image_embed' が最終的な埋め込み、'hidden_states.X' が中間層と仮定
            var imageEmbedsNode = encoderResults.FirstOrDefault(v => v.Name == EncoderOutputName);
            if (imageEmbedsNode != null)
                imageEmbeddings = imageEmbedsNode.AsTensor<float>();
            else
                throw new Exception($"Encoder output '{EncoderOutputName}' not found. Check encoder model outputs.");


            var hrFeatsNode0 = encoderResults.FirstOrDefault(v => v.Name == DecoderOutputNameHighResFeats_0);
            if (hrFeatsNode0 != null)
                highResFeats0Value = hrFeatsNode0.AsTensor<float>();
            else
                Console.WriteLine($"Warning: Encoder output '{DecoderOutputNameHighResFeats_0}' not found. Using null. Check model if this feature is required.");
            // throw new Exception($"Encoder output '{DecoderOutputNameHighResFeats_0}' not found. Check encoder model outputs.");

            var hrFeatsNode1 = encoderResults.FirstOrDefault(v => v.Name == DecoderOutputNameHighResFeats_1);
            if (hrFeatsNode1 != null)
                highResFeats1Value = hrFeatsNode1.AsTensor<float>();
            else
                Console.WriteLine($"Warning: Encoder output '{DecoderOutputNameHighResFeats_1}' not found. Using null. Check model if this feature is required.");
            // throw new Exception($"Encoder output '{DecoderOutputNameHighResFeats_1}' not found. Check encoder model outputs.");


            var allMasks = new List<SegmentationResult>();
            var pointCoordsList = points; // モデル入力サイズ基準

            int pointsPerBatch = 8;

            for (int i = 0; i < pointCoordsList.Count; i += pointsPerBatch)
            {
                var batchPoints = pointCoordsList.Skip(i).Take(pointsPerBatch).ToList();
                if (!batchPoints.Any()) continue;

                // (バッチサイズ, ポイント数/画像, 2)
                var pointCoordsTensor = new DenseTensor<float>(batchPoints.SelectMany(p => new float[] { p.X, p.Y }).ToArray(),
                                                               new int[] { batchPoints.Count, 1, 2 });
                // (バッチサイズ, ポイント数/画像)
                var pointLabelsTensor = new DenseTensor<float>(batchPoints.Select(_ => 1f).ToArray(), // 前景ポイント
                                                               new int[] { batchPoints.Count, 1 });

                // SAM 2 のデコーダは、入力として image_embeddings, point_coords, point_labels を主に取ります。
                // mask_input や has_mask_input はオプションで、通常はゼロショット推論では不要か、
                // モデルが期待するダミー値（例：ゼロテンソル）を設定します。
                // ONNXモデルの仕様を確認してください。
                // (バッチサイズ, 1, 256, 256) のダミーマスク入力 (SAMオリジナルに合わせた形状)
                var dummyMaskInput = new DenseTensor<float>(new float[batchPoints.Count * 1 * 256 * 256], new int[] { batchPoints.Count, 1, 256, 256 });
                // (バッチサイズ) のダミーhas_mask入力
                var dummyHasMaskInput = new DenseTensor<float>(Enumerable.Repeat(0f, batchPoints.Count).ToArray(), new int[] { batchPoints.Count });


                var decoderInputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(DecoderEmbeddingInputName, imageEmbeddings), // エンコーダからの画像埋め込み
                NamedOnnxValue.CreateFromTensor(DecoderPointCoordsInputName, pointCoordsTensor),
                NamedOnnxValue.CreateFromTensor(DecoderPointLabelsInputName, pointLabelsTensor),
            };

                // 高解像度特徴量をデコーダ入力に追加 (モデルが要求する場合)
                if (highResFeats0Value != null && _maskDecoderSession.InputMetadata.ContainsKey(DecoderOutputNameHighResFeats_0))
                {
                    // imageEmbeddings のバッチサイズに合わせてブロードキャストまたはリピートが必要な場合がある
                    decoderInputs.Add(NamedOnnxValue.CreateFromTensor(DecoderOutputNameHighResFeats_0, highResFeats0Value));
                }
                if (highResFeats1Value != null && _maskDecoderSession.InputMetadata.ContainsKey(DecoderOutputNameHighResFeats_1))
                {
                    decoderInputs.Add(NamedOnnxValue.CreateFromTensor(DecoderOutputNameHighResFeats_1, highResFeats1Value));
                }


                using var decoderResults = _maskDecoderSession.Run(decoderInputs);
                var masksTensor = decoderResults.First(v => v.Name == DecoderOutputMasksName).AsTensor<float>(); // (バッチ, Numマスク/ポイント, H, W)
                var iouScoresTensor = decoderResults.First(v => v.Name == DecoderOutputIouScoresName).AsTensor<float>(); // (バッチ, Numマスク/ポイント)

                var processedMasks = ProcessMasks(masksTensor, iouScoresTensor, originalWidth, originalHeight, batchPoints, predIouThresh, minMaskRegionArea);
                allMasks.AddRange(processedMasks);
            }
            return allMasks;
        }
        public List<SegmentationResult> GenerateMasks(SKBitmap image, int pointsPerSide = 32, float predIouThresh = 0.88f, int minMaskRegionArea = 0) // Image<Rgb24> を SKBitmap に変更
        {
            var pointCoordsList = GeneratePointGrids(1024, 1024, pointsPerSide); // モデル入力サイズ基準
            return GenerateMasks(image, pointCoordsList, predIouThresh, minMaskRegionArea);
        }

        private DenseTensor<float> PreprocessImage(SKBitmap image, int targetLength) // Image<Rgb24> を SKBitmap に変更
        {
            int originalWidth = image.Width;
            int originalHeight = image.Height;

            float scale = targetLength / (float)Math.Max(originalWidth, originalHeight);
            int newWidth = (int)(originalWidth * scale);
            int newHeight = (int)(originalHeight * scale);

            // リサイズ (SkiaSharp は新しい SKBitmap を返す)
            // SKFilterQuality.High は Bicubic に近い高品質なリサイズ
            using SKBitmap resizedImage = image.Resize(new SKSizeI(newWidth, newHeight), new SKSamplingOptions(SKFilterMode.Nearest, SKMipmapMode.Nearest));

            // パディングして targetLength x targetLength にする
            // SKBitmap はデフォルトで SKColorType.Bgra8888 またはプラットフォーム依存。Rgb888x を明示。
            // アルファチャンネルが不要であれば Opaque
            var paddedBitmap = new SKBitmap(targetLength, targetLength, SKColorType.Rgb888x, SKAlphaType.Opaque);
            using (var canvas = new SKCanvas(paddedBitmap))
            {
                canvas.Clear(SKColors.Black); // パディング色でクリア
                int padX = (targetLength - newWidth) / 2;
                int padY = (targetLength - newHeight) / 2;
                canvas.DrawBitmap(resizedImage, SKRect.Create(padX, padY, newWidth, newHeight));
            }

            var mean = new[] { 123.675f, 116.28f, 103.53f }; // ImageNet mean
            var std = new[] { 58.395f, 57.12f, 57.375f };   // ImageNet std

            var input = new DenseTensor<float>(new[] { 1, 3, targetLength, targetLength });
            for (int y = 0; y < targetLength; y++)
            {
                for (int x = 0; x < targetLength; x++)
                {
                    SKColor pixel = paddedBitmap.GetPixel(x, y); // SKBitmap からピクセル取得
                    input[0, 0, y, x] = (pixel.Red - mean[0]) / std[0];
                    input[0, 1, y, x] = (pixel.Green - mean[1]) / std[1];
                    input[0, 2, y, x] = (pixel.Blue - mean[2]) / std[2];
                }
            }
            paddedBitmap.Dispose(); // 使用後に破棄
            return input;
        }

        private List<SKPoint> GeneratePointGrids(int imageWidth, int imageHeight, int pointsPerSide) // PointF を SKPoint に変更
        {
            var points = new List<SKPoint>();
            if (pointsPerSide <= 0) return points;

            float stepX = imageWidth / (float)pointsPerSide;
            float stepY = imageHeight / (float)pointsPerSide;

            for (int i = 0; i < pointsPerSide; i++)
            {
                for (int j = 0; j < pointsPerSide; j++)
                {
                    points.Add(new SKPoint((j + 0.5f) * stepX, (i + 0.5f) * stepY));
                }
            }
            return points;
        }
        private List<SegmentationResult> ProcessMasks(
            Tensor<float> masksTensor,
            Tensor<float> iouScoresTensor,
            int originalImageWidth, // 元の画像の幅と高さ
            int originalImageHeight,
            List<SKPoint> batchPoints,
            float predIouThresh,
            int minMaskRegionArea) // PointF を SKPoint に変更
        {
            var results = new List<SegmentationResult>();

            int batchSize = masksTensor.Dimensions[0];
            // int numMasksPerPoint = masksTensor.Dimensions[1]; // 通常 SAM はポイントごとに複数のマスクを出力しうる
            int numMasksPerPoint = Math.Min(masksTensor.Dimensions[1], iouScoresTensor.Dimensions[1]); // 安全のため小さい方
            int maskHeight = masksTensor.Dimensions[2]; // モデルの出力マスクの高さ
            int maskWidth = masksTensor.Dimensions[3];  // モデルの出力マスクの幅

            for (int i = 0; i < batchSize; i++)
            {
                for (int j = 0; j < numMasksPerPoint; j++)
                {
                    float iouScore = iouScoresTensor[i, j];
                    if (iouScore < predIouThresh) continue;

                    var mask = new float[maskHeight, maskWidth];
                    for (int y = 0; y < maskHeight; y++)
                    {
                        for (int x = 0; x < maskWidth; x++)
                        {
                            mask[y, x] = masksTensor[i, j, y, x];
                        }
                    }
                    bool[,] binaryMask = new bool[maskHeight, maskWidth];
                    int minX = maskWidth, minY = maskHeight, maxX = -1, maxY = -1;
                    int area = 0;

                    for (int y = 0; y < maskHeight; y++)
                    {
                        for (int x = 0; x < maskWidth; x++)
                        {
                            if (mask[y, x] > 0.0f)
                            {
                                binaryMask[y, x] = true;
                                area++;
                                if (x < minX) minX = x;
                                if (x > maxX) maxX = x;
                                if (y < minY) minY = y;
                                if (y > maxY) maxY = y;
                            }
                            else
                            {
                                binaryMask[y, x] = false;
                            }
                        }
                    }

                    if (area < minMaskRegionArea || minX > maxX || minY > maxY) // 小さすぎる領域や無効な領域はスキップ
                        continue;

                    float inputTargetLength = (float)_imageEncoderSession.InputMetadata[EncoderInputName].Dimensions[2]; // 例: 1024
                    float scaleToOrigW = (float)originalImageWidth / inputTargetLength;
                    float scaleToOrigH = (float)originalImageHeight / inputTargetLength;


                    float boxScaleX = (float)originalImageWidth / maskWidth;
                    float boxScaleY = (float)originalImageHeight / maskHeight;


                    var result = new SegmentationResult
                    {
                        // BoundingBoxは元の画像座標系に変換
                        BoundingBox = SKRectI.Create(
                            (int)(minX * boxScaleX),
                            (int)(minY * boxScaleY),
                            (int)((maxX - minX + 1) * boxScaleX),
                            (int)((maxY - minY + 1) * boxScaleY)
                        ),
                        Mask = binaryMask,
                        PredictedIou = iouScore,
                        PointCoords = new List<SKPoint> { batchPoints[i] }
                    };
                    results.Add(result);
                }
            }
            return results;
        }
    }

    public struct SegmentationResult
    {
        public bool[,] Mask { get; set; } // (Height, Width) - モデル出力解像度
        public float PredictedIou { get; set; }
        public float StabilityScore { get; set; }
        public SKRectI BoundingBox { get; set; } // (X, Y, Width, Height) - 元画像座標系
        public int Area => Mask?.Cast<bool>().Count(isSet => isSet) ?? 0; // モデル出力解像度でのピクセル単位の面積
        public List<SKPoint> PointCoords { get; set; }
    }

}
