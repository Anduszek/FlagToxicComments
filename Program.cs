using System;
using System.ComponentModel.DataAnnotations.Schema;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.VisualBasic.FileIO;

namespace FlagToxicComments
{
    public class TrainingInput
    {
        [LoadColumn(0)]
        public string Id { get; set; }

        [LoadColumn(1)]
        public string CommentText { get; set; }

        [LoadColumn(2)]
        public bool Toxic { get; set; }

        [LoadColumn(3)]
        public bool SevereToxic { get; set; }

        [LoadColumn(4)]
        public bool Obscene { get; set; }

        [LoadColumn(5)]
        public bool Threat { get; set; }

        [LoadColumn(6)]
        public bool Insult { get; set; }

        [LoadColumn(7)]
        public bool IdentityHate { get; set; }
    }

    public class ToLabel
    {
        public float Rating { get; set; }
    }

    public class TestInput
    {
        public string Id { get; set; }
        public string CommentText { get; set; }
    }

    public class BinaryTestPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Score { get; set; }
    }

    public class MulitClassTestPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint Rating { get; set; }

        //[ColumnName("Score")]
        //public float[] Score { get; set; }

        //public float Rating { get; set; }
    }

    class Program
    {
        static void Main(
            string[] args)
        {
            var context = new MLContext();
            Console.Write("Loading data...");
            var data = context.Data.LoadFromTextFile<TrainingInput>(
                Path.Combine(Environment.CurrentDirectory, "train.tsv"),
                hasHeader: true);

            Console.WriteLine("Done");

            var partitions = context.Data.TrainTestSplit(
                data,
                testFraction: 0.2);

            TrainAndTestWithMultipleBinaryClassificationPipelines(context, partitions);
            TrainAndTestWithMultiClassificationPipeline(context, partitions);

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }

        private static void TrainAndTestWithMultipleBinaryClassificationPipelines(
            MLContext context,
            DataOperationsCatalog.TrainTestData partitions)
        {
            var initialPipeline =
                context.Transforms.Text.FeaturizeText(
                        outputColumnName: "Features",
                        inputColumnName: "CommentText")
                    .AppendCacheCheckpoint(context);

            var toxicModel = BuildAndTrainBinaryModel(context, partitions, initialPipeline, "Toxic");
            var severeToxicModel = BuildAndTrainBinaryModel(context, partitions, initialPipeline, "SevereToxic");
            var obsceneModel = BuildAndTrainBinaryModel(context, partitions, initialPipeline, "Obscene");
            var threatModel = BuildAndTrainBinaryModel(context, partitions, initialPipeline, "Threat");
            var insultModel = BuildAndTrainBinaryModel(context, partitions, initialPipeline, "Insult");
            var identityHateModel = BuildAndTrainBinaryModel(context, partitions, initialPipeline, "IdentityHate");
            using var testFileStream = File.OpenRead(Path.Combine(Environment.CurrentDirectory, "test.tsv"));
            var parser = new TextFieldParser(testFileStream)
            {
                Delimiters = new[] {"\t"},
                TextFieldType = FieldType.Delimited
            };

            var firstLine = true;
            var toxicCount = 0;
            var severeToxicCount = 0;
            var obsceneCount = 0;
            var threatCount = 0;
            var insultCount = 0;
            var identityHateCount = 0;

            while (!parser.EndOfData)
            {
                var values = parser.ReadFields();
                if (firstLine)
                {
                    firstLine = false;
                    continue;
                }

                var testInput = new TestInput
                {
                    Id = values[0],
                    CommentText = values[1]
                };
                
                var isToxic = toxicModel.Predict(testInput).Score;
                var isSevereToxic = severeToxicModel.Predict(testInput).Score;
                var isObscene = obsceneModel.Predict(testInput).Score;
                var isThreat = threatModel.Predict(testInput).Score;
                var isInsult = insultModel.Predict(testInput).Score;
                var isIdentityHate = identityHateModel.Predict(testInput).Score;

                toxicCount += isToxic ? 1 : 0;
                severeToxicCount += isSevereToxic ? 1 : 0;
                obsceneCount += isObscene ? 1 : 0;
                threatCount += isThreat ? 1 : 0;
                insultCount += isInsult ? 1 : 0;
                identityHateCount += isIdentityHate ? 1 : 0;

                //if (isToxic || isSevereToxic || isObscene || isThreat || isInsult || isIdentityHate)
                //{
                //    Console.WriteLine($"{testInput.CommentText} - ");
                //    Console.WriteLine();
                //    Console.WriteLine($"Toxic: {isToxic}, " +
                //                      $"Severe Toxic: {isSevereToxic}, " +
                //                      $"Obscene: {isObscene}, " +
                //                      $"Threat: {isThreat}, " +
                //                      $"Insult: {isInsult}, " +
                //                      $"Identity Hate: {isIdentityHate}");
                //    break;
                //}
            }

            Console.WriteLine($"Toxic: {toxicCount}, " +
                              $"Severe Toxic: {severeToxicCount}, " +
                              $"Obscene: {obsceneCount}, " +
                              $"Threat: {threatCount}, " +
                              $"Insult: {insultCount}, " +
                              $"Identity Hate: {identityHateCount}");
        }

        private static PredictionEngine<TestInput, BinaryTestPrediction> BuildAndTrainBinaryModel(
            MLContext context,
            DataOperationsCatalog.TrainTestData partitions,
            IEstimator<ITransformer> pipeline,
            string labelColumnName)
        {
            Console.WriteLine($"Training for {labelColumnName}");

            var modelPipeline = pipeline.Append(context.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: labelColumnName));
            var model = modelPipeline.Fit(partitions.TrainSet);
            var predictions = model.Transform(partitions.TestSet);
            var metrics = context.BinaryClassification.Evaluate(
                data: predictions,
                labelColumnName: labelColumnName,
                scoreColumnName: "Score");
            Console.WriteLine(labelColumnName);
            Console.WriteLine($"  Accuracy:          {metrics.Accuracy:P2}");
            Console.WriteLine($"  Auc:               {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"  Auprc:             {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"  F1Score:           {metrics.F1Score:P2}");
            Console.WriteLine($"  LogLoss:           {metrics.LogLoss:0.##}");
            Console.WriteLine($"  LogLossReduction:  {metrics.LogLossReduction:0.##}");
            Console.WriteLine($"  PositivePrecision: {metrics.PositivePrecision:0.##}");
            Console.WriteLine($"  PositiveRecall:    {metrics.PositiveRecall:0.##}");
            Console.WriteLine($"  NegativePrecision: {metrics.NegativePrecision:0.##}");
            Console.WriteLine($"  NegativeRecall:    {metrics.NegativeRecall:0.##}");
            Console.WriteLine();


            return context.Model.CreatePredictionEngine<TestInput, BinaryTestPrediction>(model);
        }


        private static void TrainAndTestWithMultiClassificationPipeline(
            MLContext context,
            DataOperationsCatalog.TrainTestData partitions)
        {
            var pipeline =
                context.Transforms.CustomMapping<TrainingInput, ToLabel>(
                        (
                            input,
                            output) =>
                        {
                            var rating = 0;
                            if (input.Toxic)
                            {
                                rating += 1 << 6;
                            }

                            if (input.SevereToxic)
                            {
                                rating += 1 << 5;
                            }

                            if (input.Obscene)
                            {
                                rating += 1 << 4;
                            }

                            if (input.Threat)
                            {
                                rating += 1 << 3;
                            }

                            if (input.Insult)
                            {
                                rating += 1 << 2;
                            }

                            if (input.IdentityHate)
                            {
                                rating += 1 << 1;
                            }

                            output.Rating = rating;
                        },
                        "LabelMapping"
                    )
                    .Append(
                        context.Transforms.Conversion.MapValueToKey(
                            outputColumnName: "Label",
                            inputColumnName: "Rating",
                            keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                    )
                    .Append(
                        context.Transforms.Text.FeaturizeText(
                            outputColumnName: "Features",
                            inputColumnName: "CommentText"))
                    .AppendCacheCheckpoint(context)
                    .Append(context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(
                        labelColumnName: "Label",
                        featureColumnName: "Features"))
                    .Append(context.Transforms.Conversion.MapKeyToValue(
                        outputColumnName: "Rating",
                        inputColumnName: "Label"));

            Console.Write("Training model...");
            var model =
                pipeline.Fit(partitions.TrainSet);

            Console.WriteLine("Done");

            Console.WriteLine("Evaluating model");
            var predictions = model.Transform(partitions.TestSet);
            var metrics = context.MulticlassClassification.Evaluate(
                data: predictions,
                labelColumnName: "Label",
                scoreColumnName: "Score");

            Console.WriteLine($"Evaluation metrics");
            Console.WriteLine($"    MicroAccuracy:    {metrics.MicroAccuracy:0.###}");
            Console.WriteLine($"    MacroAccuracy:    {metrics.MacroAccuracy:0.###}");
            Console.WriteLine($"    LogLoss:          {metrics.LogLoss:#.###}");
            Console.WriteLine($"    LogLossReduction: {metrics.LogLossReduction:#.###}");
            Console.WriteLine();

            var predictionEngine = context.Model.CreatePredictionEngine<TrainingInput, MulitClassTestPrediction>(model);

            using var testFileStream = File.OpenRead(Path.Combine(Environment.CurrentDirectory, "test.tsv"));
            var parser = new TextFieldParser(testFileStream)
            {
                Delimiters = new[] {"\t"},
                TextFieldType = FieldType.Delimited
            };

            var firstLine = true;
            var toxicCount = 0;
            var severeToxicCount = 0;
            var obsceneCount = 0;
            var threatCount = 0;
            var insultCount = 0;
            var identityHateCount = 0;
            
            while (!parser.EndOfData)
            {
                var values = parser.ReadFields();
                if (firstLine)
                {
                    firstLine = false;
                    continue;
                }

                var testInput = new TrainingInput
                {
                    Id = values[0],
                    CommentText = values[1]
                };


                var prediction = predictionEngine.Predict(testInput);

                var isToxic = ((int) prediction.Rating & (1 << 6)) != 0;
                var isSevereToxic = ((int) prediction.Rating & (1 << 5)) != 0;
                var isObscene = ((int) prediction.Rating & (1 << 4)) != 0;
                var isThreat = ((int) prediction.Rating & (1 << 3)) != 0;
                var isInsult = ((int) prediction.Rating & (1 << 2)) != 0;
                var isIdentityHate = ((int) prediction.Rating & (1 << 1)) != 0;

                toxicCount += isToxic ? 1 : 0;
                severeToxicCount += isSevereToxic ? 1 : 0;
                obsceneCount += isObscene ? 1 : 0;
                threatCount += isThreat ? 1 : 0;
                insultCount += isInsult ? 1 : 0;
                identityHateCount += isIdentityHate ? 1 : 0;

                //if (isToxic || isSevereToxic || isObscene || isThreat || isInsult || isIdentityHate)
                //{
                //    Console.WriteLine($"{testInput.CommentText} - ");
                //    Console.WriteLine();
                //    Console.WriteLine($"Toxic: {isToxic}, " +
                //                      $"Severe Toxic: {isSevereToxic}, " +
                //                      $"Obscene: {isObscene}, " +
                //                      $"Threat: {isThreat}, " +
                //                      $"Insult: {isInsult}, " +
                //                      $"Identity Hate: {isIdentityHate}");
                //    Console.WriteLine();
                //}
            }

            Console.WriteLine($"Toxic: {toxicCount}, " +
                              $"Severe Toxic: {severeToxicCount}, " +
                              $"Obscene: {obsceneCount}, " +
                              $"Threat: {threatCount}, " +
                              $"Insult: {insultCount}, " +
                              $"Identity Hate: {identityHateCount}");
        }
    }
}
