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
    private ObservableCollection<ObservablePoint> _trainingDataPoints;
    [ObservableProperty]
    private ObservableCollection<ObservablePoint> _predictedDataPoints;
    [ObservableProperty]
    private ObservableCollection<ObservablePoint> _lossesDataPoints;

    [ObservableProperty]
    private int _numberOfDataPoints = 20;
    [ObservableProperty]
    private double _dataRange = 0.1;

    [ObservableProperty]
    private double _learningRate = 0.05;
    [ObservableProperty]
    private int _epochCount = 500;

    [ObservableProperty]
    private double _calculatedMSE;

    private TorchSharp.Modules.Sequential _ann;
    private torch.Tensor _xTensor;
    private torch.Tensor _yTensor;
    private torch.Tensor _testLoss;

    private readonly int _decimals = 2;

    public MainViewModel()
    {
        _trainingDataPoints = [];
        _predictedDataPoints = [];
        _lossesDataPoints = [];

        _ann = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Linear(1, 1));
        _xTensor = torch.randn(0, 0);
        _yTensor = torch.randn(0, 0);
        _testLoss = torch.randn(0, 0);

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

        _dataPointsXAxes =
        [
            new Axis
            {
                MaxLimit = 1,
                MinLimit =  - 1,
                LabelsPaint = new SolidColorPaint(SKColors.LightGray)
            }
        ];
        _dataPointsYAxes =
        [
            new Axis
            {
                MaxLimit =  1,
                MinLimit =  - 1,
                LabelsPaint = new SolidColorPaint(SKColors.LightGray)
            }
        ];

        _lossesPointsSeries = new ISeries[1];
        _lossesPointsSeries[0] = new ScatterSeries<ObservablePoint>
        {
            Name = "Losses",
            Fill = new SolidColorPaint(SKColors.Red),
            GeometrySize = 6,
            Values = _lossesDataPoints
        };

        _lossesPointsXAxes =
        [
            new Axis
            {
                MaxLimit = EpochCount,
                MinLimit =  0,
                LabelsPaint = new SolidColorPaint(SKColors.LightGray)
            }
        ];
        _lossesPointsYAxes =
        [
            new Axis
            {
                MaxLimit =  1,
                MinLimit =  0,
                LabelsPaint = new SolidColorPaint(SKColors.LightGray)
            }
        ];
    }

    [RelayCommand]
    private void ResetModel()
    {
        _ann = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Linear(1, 1));
    }

    [RelayCommand]
    private void RandomizeTrainingData()
    {
        _xTensor = torch.randn(NumberOfDataPoints, 1);
        _yTensor = _xTensor + torch.randn(NumberOfDataPoints, 1) * DataRange;

        var observablePoints = GetObservablePointsFromTensors(_xTensor, _yTensor, _decimals);

        TrainingDataPoints.Clear();
        foreach (var observablePoint in observablePoints)
        {
            TrainingDataPoints.Add(observablePoint);
        }
    }

    [RelayCommand]
    private void TrainModel()
    {
        if (_xTensor.shape[0] == 0)
            return;

        var lossFunction = torch.nn.MSELoss();

        var optimizer = torch.optim.SGD(_ann.parameters(), LearningRate);

        var xTensor = torch.arange(1.0, EpochCount);
        var losses = torch.zeros(EpochCount);

        for (var epoch = 0; epoch < EpochCount; epoch++)
        {
            var yHat = _ann.forward(_xTensor);

            var loss = lossFunction.forward(yHat, _yTensor);

            losses[epoch] = loss.item<float>();

            optimizer.zero_grad();

            loss.backward();

            optimizer.step();
        }

        _ann.zero_grad();

        var observablePoints = GetObservablePointsFromTensors(xTensor, losses, _decimals);
        foreach (var observablePoint in observablePoints)
        {
            LossesDataPoints.Add(observablePoint);
        }
    }

    [RelayCommand]
    private void PredictData()
    {
        if (_xTensor.shape[0] == 0)
            return;

        var predictedY = _ann.forward(_xTensor);

        var testLoss = (predictedY - _yTensor).pow(2).mean();
        CalculatedMSE = testLoss.item<float>();

        var observablePoints = GetObservablePointsFromTensors(_xTensor, predictedY, _decimals);

        PredictedDataPoints.Clear();
        foreach (var observablePoint in observablePoints)
        {
            PredictedDataPoints.Add(observablePoint);
        }
    }

    private static ObservableCollection<ObservablePoint> GetObservablePointsFromTensors(torch.Tensor xTensor, torch.Tensor yTensor, int decimals)
    {
        var observablePoints = new ObservableCollection<ObservablePoint>();

        if (!(xTensor.shape.Length == yTensor.shape.Length))
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