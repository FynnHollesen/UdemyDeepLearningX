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
using static TorchSharp.torch.nn;

namespace UdemyDeepLearningX.ViewModels;

public partial class MainViewModel : ViewModelBase
{
    [ObservableProperty]
    private ISeries[] _series;

    [ObservableProperty]
    private Axis[] _xAxes;

    [ObservableProperty]
    private Axis[] _yAxes;

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

        RandomizeChart();
    }

    [RelayCommand]
    private void RandomizeChart()
    {
        var valueCount = 30;
        var x = torch.randn(valueCount, 1);
        var y = x + torch.randn(valueCount, 1) / 2;

        var observablePoints = new ObservableCollection<ObservablePoint>();

        for (var i = 0; i < valueCount; i++)
        {
            observablePoints.Add(new ObservablePoint(x[i].item<float>() * 100, y[i].item<float>() * 100));
        }

        Series = new ISeries[1];
        Series[0] = new ScatterSeries<ObservablePoint>
        {
            Values = observablePoints,
            Fill = new SolidColorPaint(SKColors.LightBlue),
            GeometrySize = 12
        };
    }
}