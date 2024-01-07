using System;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using LiveChartsCore;
using LiveChartsCore.Defaults;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using SkiaSharp;

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

        Series = Array.Empty<ISeries>();

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
        var random = new Random();
        var values = new ObservableCollection<ObservablePoint>();

        for (var i = 0; i < 30; i++)
        {
            var value = random.Next(-100, 100);
            values.Add(new ObservablePoint(value, value + random.Next(-50, 50)));
        }

        Series = new ISeries[1];
        Series[0] = new ScatterSeries<ObservablePoint>
        {
            Values = values,
            Fill = new SolidColorPaint(SKColors.LightBlue),
            GeometrySize = 12
        };
    }
}