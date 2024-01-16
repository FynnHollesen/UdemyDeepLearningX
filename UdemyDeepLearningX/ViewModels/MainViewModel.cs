using System;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LiveChartsCore;
using LiveChartsCore.Defaults;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using SkiaSharp;
using TorchSharp;

namespace UdemyDeepLearningX.ViewModels;

public partial class MainViewModel : ViewModelBase
{
    [ObservableProperty]
    private ISeries[] _dataPointsSeries;
    [ObservableProperty]
    private Axis[] _dataPointsXAxes;
    [ObservableProperty]
    private Axis[] _dataPointsYAxes;

    [ObservableProperty]
    private ISeries[] _lossesPointsSeries;
    [ObservableProperty]
    private Axis[] _lossesPointsXAxes;
    [ObservableProperty]
    private Axis[] _lossesPointsYAxes;

    [ObservableProperty]
    private ISeries[] _costPointsSeries;
    [ObservableProperty]
    private Axis[] _costPointsXAxes;
    [ObservableProperty]
    private Axis[] _costPointsYAxes;

    [ObservableProperty]
    private ObservableCollection<ObservablePoint> _trainingDataPoints;
    [ObservableProperty]
    private ObservableCollection<ObservablePoint> _predictedDataPoints;
    [ObservableProperty]
    private ObservableCollection<ObservablePoint> _lossesDataPoints;
    [ObservableProperty]
    private ObservableCollection<ObservablePoint> _costDataPoints;

    [ObservableProperty]
    private int _numberOfDataPoints = 20;
    [ObservableProperty]
    private double _dataRange = 0.5;
    [ObservableProperty]
    private double _dataSlope = 1;
    [ObservableProperty]
    private int _dataExponent = 1;
    [ObservableProperty]
    private double _learningRate = 0.05;
    [ObservableProperty]
    private int _epochCount = 500;
    [ObservableProperty]
    private int _trainingIterations = 10;

    private TorchSharp.Modules.Sequential _ann;
    private torch.Tensor _dataPointsXTensor;
    private torch.Tensor _dataPointsYTensor;
    private torch.Tensor _costTensor;
    private int _currentTrainingIteration = 0;

    private readonly int _decimals = 2;

    public MainViewModel()
    {
        _trainingDataPoints = [];
        _predictedDataPoints = [];
        _lossesDataPoints = [];
        _costDataPoints = [];

        _ann = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Linear(1, 1));
        _dataPointsXTensor = torch.randn(0, 0);
        _dataPointsYTensor = torch.randn(0, 0);
        _costTensor = torch.randn(0, 0);

        _dataPointsSeries = new ISeries[2];
        _dataPointsSeries[0] = new ScatterSeries<ObservablePoint>
        {
            Name = "Training data",
            Fill = new SolidColorPaint(SKColors.LightBlue),
            GeometrySize = 12,
            Values = _trainingDataPoints
        };

        _dataPointsSeries[1] = new ScatterSeries<ObservablePoint>
        {
            Name = "Predicted data",
            Fill = new SolidColorPaint(SKColors.PaleVioletRed),
            GeometrySize = 12,
            Values = _predictedDataPoints
        };

        _lossesPointsSeries = new ISeries[1];
        _lossesPointsSeries[0] = new LineSeries<ObservablePoint>
        {
            Name = "Losses",
            Fill = null,
            GeometrySize = 6,
            GeometryStroke = new SolidColorPaint(SKColors.PaleVioletRed),
            Stroke = new SolidColorPaint(SKColors.PaleVioletRed),
            Values = _lossesDataPoints
        };

        _costPointsSeries = new ISeries[1];
        _costPointsSeries[0] = new LineSeries<ObservablePoint>
        {
            Name = "Cost",
            Fill = null,
            GeometrySize = 6,
            GeometryStroke = new SolidColorPaint(SKColors.PaleVioletRed),
            Stroke = new SolidColorPaint(SKColors.PaleVioletRed),
            Values = _costDataPoints
        };

        _dataPointsXAxes =
            [
            new Axis
            {
                MaxLimit = 1,
                MinLimit =  - 1,
                LabelsPaint = new SolidColorPaint(SKColors.LightGray)
            }];
        _dataPointsYAxes =
        [
            new Axis
            {
                MaxLimit =  1,
                MinLimit =  - 1,
                LabelsPaint = new SolidColorPaint(SKColors.LightGray)
            }
        ];
        _lossesPointsXAxes =
        [
            new Axis
            {
                MinLimit = 1,
                LabelsPaint = new SolidColorPaint(SKColors.LightGray)
            }
        ];
        _lossesPointsYAxes =
        [
            new Axis
            {
                MinLimit = 0,
                LabelsPaint = new SolidColorPaint(SKColors.LightGray)
            }
        ];
        _costPointsXAxes =
        [
            new Axis
            {
                MinLimit = 1,
                LabelsPaint = new SolidColorPaint(SKColors.LightGray)
            }
        ];
        _costPointsYAxes =
        [
            new Axis
            {
                MinLimit = 0,
                LabelsPaint = new SolidColorPaint(SKColors.LightGray)
            }
        ];
    }

    [RelayCommand]
    private void ResetModel()
    {
        _ann = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Linear(1, 1));
        PredictedDataPoints.Clear();
        LossesDataPoints.Clear();
        CostDataPoints.Clear();
    }

    [RelayCommand]
    private void RandomizeTrainingData()
    {
        _dataPointsXTensor = torch.randn(NumberOfDataPoints, 1);
        _dataPointsYTensor = DataSlope * torch.pow(_dataPointsXTensor, DataExponent) + torch.randn(NumberOfDataPoints, 1) * DataRange;

        var observablePoints = GetObservablePointsFromTensors(_dataPointsXTensor, _dataPointsYTensor, _decimals);

        TrainingDataPoints.Clear();
        foreach (var observablePoint in observablePoints)
        {
            TrainingDataPoints.Add(observablePoint);
        }
    }

    [RelayCommand]
    private void TrainModel()
    {
        if (_currentTrainingIteration == 0)
        {
            CostDataPoints.Clear();
            _costTensor = torch.zeros(TrainingIterations);
        }

        if (_dataPointsXTensor.shape[0] == 0)
        {
            RandomizeTrainingData();
        }

        var lossFunction = torch.nn.MSELoss();

        var optimizer = torch.optim.SGD(_ann.parameters(), LearningRate);

        var losses = torch.zeros(EpochCount);

        for (var epoch = 0; epoch < EpochCount; epoch++)
        {
            var yHat = _ann.forward(_dataPointsXTensor);

            var loss = lossFunction.forward(yHat, _dataPointsYTensor);

            losses[epoch] = loss.item<float>();

            optimizer.zero_grad();

            loss.backward();

            optimizer.step();
        }

        _ann.zero_grad();

        var cost = lossFunction.forward(losses, torch.zeros(TrainingIterations, 1));
        _costTensor[_currentTrainingIteration] = cost.item<float>();

        _currentTrainingIteration++;

        if (_currentTrainingIteration == TrainingIterations)
        {
            var lossesXTensor = torch.arange(1.0, EpochCount + 1);
            var observableLossesPoints = GetObservablePointsFromTensors(lossesXTensor, losses, _decimals);
            LossesDataPoints.Clear();
            foreach (var observablePoint in observableLossesPoints)
            {
                LossesDataPoints.Add(observablePoint);
            }

            var costXTensor = torch.arange(1.0, TrainingIterations + 1);
            var observableCostPoints = GetObservablePointsFromTensors(costXTensor, _costTensor, _decimals);
            foreach (var observableCostPoint in observableCostPoints)
            {
                CostDataPoints.Add(observableCostPoint);
            }

            PredictData();

            _currentTrainingIteration = 0;
            return;
        }

        RandomizeTrainingData();
        TrainModel();
    }

    [RelayCommand]
    private void PredictData()
    {
        if (_dataPointsXTensor.shape[0] == 0)
            return;

        var predictedY = _ann.forward(_dataPointsXTensor);

        var observablePoints = GetObservablePointsFromTensors(_dataPointsXTensor, predictedY, _decimals);

        PredictedDataPoints.Clear();
        foreach (var observablePoint in observablePoints)
        {
            PredictedDataPoints.Add(observablePoint);
        }
    }

    private static ObservableCollection<ObservablePoint> GetObservablePointsFromTensors(torch.Tensor xTensor, torch.Tensor yTensor, int decimals)
    {
        var observablePoints = new ObservableCollection<ObservablePoint>();

        if (xTensor.shape.Length == 0 || (xTensor.shape.Length != yTensor.shape.Length))
            return observablePoints;

        for (var i = 0; i < xTensor.shape[0]; i++)
        {
            var xPosition = Math.Round(xTensor[i].item<float>(), decimals);
            var yPosition = Math.Round(yTensor[i].item<float>(), decimals);
            observablePoints.Add(new ObservablePoint(xPosition, yPosition));
        }

        return observablePoints;
    }
}