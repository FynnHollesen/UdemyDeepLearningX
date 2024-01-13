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

    private torch.Tensor? _xTensor;
    private torch.Tensor? _yTensor;

    public MainViewModel()
    {

        Series = [];

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

        RandomizeValues();
    }

    [RelayCommand]
    private void RandomizeValues()
    {
        var valueCount = 30;
        _xTensor = torch.randn(valueCount, 1);
        _yTensor = _xTensor + torch.randn(valueCount, 1) / 2;

        var observablePoints = GetObservablePointsFromTensors(_xTensor, _yTensor, 100);

        Series = new ISeries[1];
        Series[0] = new ScatterSeries<ObservablePoint>
        {
            Values = observablePoints,
            Fill = new SolidColorPaint(SKColors.LightBlue),
            GeometrySize = 12
        };
    }

    private static ObservableCollection<ObservablePoint> GetObservablePointsFromTensors(torch.Tensor xTensor, torch.Tensor yTensor, float factor)
    {
        var observablePoints = new ObservableCollection<ObservablePoint>();

        if (!(xTensor.shape.Length == yTensor.shape.Length))
            return observablePoints;

        for (var i = 0; i < xTensor.shape[0]; i++)
        {
            observablePoints.Add(new ObservablePoint(xTensor[i].item<float>() * factor, yTensor[i].item<float>() * factor));
        }

        return observablePoints;
    }
}