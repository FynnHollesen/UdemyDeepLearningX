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
    private float _calculatedMSE;

    private torch.Tensor _xTensor;
    private torch.Tensor _yTensor;
    private torch.Tensor _testLoss;

    private readonly float _factor = 100;
    private readonly int _decimals = 2;

    public MainViewModel()
    {

        Series = new ISeries[2];
        Series[0] = new ScatterSeries<ObservablePoint>
        {
            Name = "Training data",
            Fill = new SolidColorPaint(SKColors.LightBlue),
            GeometrySize = 12
        };
        Series[1] = new ScatterSeries<ObservablePoint>
        {
            Name = "Predicted data",
            Fill = new SolidColorPaint(SKColors.PaleVioletRed),
            GeometrySize = 12
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

        RandomizeTrainingData();
        RunModel();
    }

    [RelayCommand]
    private void RandomizeTrainingData()
    {
        var valueCount = 30;
        _xTensor = torch.randn(valueCount, 1);
        _yTensor = _xTensor + torch.randn(valueCount, 1) / 2;

        var observablePoints = GetObservablePointsFromTensors(_xTensor, _yTensor, _factor, _decimals);

        Series[0].Values = observablePoints;
    }

    [RelayCommand]
    private void RunModel()
    {
        var ann = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.ReLU(), torch.nn.Linear(1, 1));

        var learningRate = 0.05;

        var lossFunction = torch.nn.MSELoss();

        var optimizer = torch.optim.SGD(ann.parameters(), learningRate);

        var epochCount = 500;
        
        var losses = torch.zeros(epochCount);

        for (var epoch = 0; epoch < epochCount; epoch++)
        {
            var yHat = ann.forward(_xTensor);

            var loss = lossFunction.forward(yHat, _yTensor);

            losses[epoch] = loss.item<float>();

            optimizer.zero_grad();

            loss.backward();

            optimizer.step();
        }

        var predictedY = ann.forward(_xTensor);

        var testLoss = (predictedY - _yTensor).pow(2).mean();
        CalculatedMSE = testLoss.item<float>();

        var observablePoints = GetObservablePointsFromTensors(_xTensor, predictedY, _factor, _decimals);

        Series[1].Values = observablePoints;
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