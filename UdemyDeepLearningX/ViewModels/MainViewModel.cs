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
    private ISeries[] _series;
    [ObservableProperty]
    private Axis[] _xAxes;
    [ObservableProperty]
    private Axis[] _yAxes;

    [ObservableProperty]
    private ObservableCollection<ObservablePoint> _trainingDataPoints;
    [ObservableProperty]
    private ObservableCollection<ObservablePoint> _predictedDataPoints;

    [ObservableProperty]
    private int _numberOfDataPoints = 20;

    [ObservableProperty]
    private float _learningRate = 0.05f;
    [ObservableProperty]
    private int _epochCount = 500;

    [ObservableProperty]
    private float _calculatedMSE;

    private TorchSharp.Modules.Sequential _ann;
    private torch.Tensor _xTensor;
    private torch.Tensor _yTensor;
    private torch.Tensor _testLoss;

    private readonly float _factor = 100;
    private readonly int _decimals = 2;

    public MainViewModel()
    {
        _trainingDataPoints = new ObservableCollection<ObservablePoint>();
        _predictedDataPoints = new ObservableCollection<ObservablePoint>();

        _ann = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Linear(1, 1));
        _xTensor = torch.randn(0, 0);
        _yTensor = torch.randn(0, 0);
        _testLoss = torch.randn(0, 0);

        Series = new ISeries[2];
        Series[0] = new ScatterSeries<ObservablePoint>
        {
            Name = "Training data",
            Fill = new SolidColorPaint(SKColors.LightBlue),
            GeometrySize = 12,
            Values = _trainingDataPoints
        };
        Series[1] = new ScatterSeries<ObservablePoint>
        {
            Name = "Predicted data",
            Fill = new SolidColorPaint(SKColors.PaleVioletRed),
            GeometrySize = 12,
            Values = _predictedDataPoints
        };

        XAxes =
        [
            new Axis
            {
                MaxLimit = 150,
                MinLimit = -150,
                LabelsPaint = new SolidColorPaint(SKColors.LightGray)
            }
        ];
        YAxes =
        [
            new Axis
            {
                MaxLimit = 150,
                MinLimit = -150,
                LabelsPaint = new SolidColorPaint(SKColors.LightGray)
            }
        ];
    }

    [RelayCommand]
    private void RandomizeTrainingData()
    {
        _xTensor = torch.randn(NumberOfDataPoints, 1);
        _yTensor = _xTensor + torch.randn(NumberOfDataPoints, 1) / 2;

        var observablePoints = GetObservablePointsFromTensors(_xTensor, _yTensor, _factor, _decimals);

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
    }

    [RelayCommand]
    private void PredictData()
    {
        if (_xTensor.shape[0] == 0)
            return;

        var predictedY = _ann.forward(_xTensor);

        var testLoss = (predictedY - _yTensor).pow(2).mean();
        CalculatedMSE = testLoss.item<float>();

        var observablePoints = GetObservablePointsFromTensors(_xTensor, predictedY, _factor, _decimals);

        PredictedDataPoints.Clear();
        foreach (var observablePoint in observablePoints)
        {
            PredictedDataPoints.Add(observablePoint);
        }
    }

    private static ObservableCollection<ObservablePoint> GetObservablePointsFromTensors(torch.Tensor xTensor, torch.Tensor yTensor, float factor, int decimals)
    {
        var observablePoints = new ObservableCollection<ObservablePoint>();

        if (!(xTensor.shape.Length == yTensor.shape.Length))
            return observablePoints;

        for (var i = 0; i < xTensor.shape[0]; i++)
        {
            var xPosition = Math.Round(xTensor[i].item<float>() * factor, decimals);
            var yPosition = Math.Round(yTensor[i].item<float>() * factor, decimals);
            observablePoints.Add(new ObservablePoint(xPosition, yPosition));
        }

        return observablePoints;
    }
}