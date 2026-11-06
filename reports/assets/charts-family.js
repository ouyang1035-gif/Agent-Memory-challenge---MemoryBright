(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();

  var el = document.getElementById('chart-family');
  if (!el || !window.echarts) return;

  var chart = echarts.init(el, null, { renderer: 'svg' });
  chart.setOption({
    animation: false,
    color: [accent, accent2],
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      formatter: function (params) {
        var out = params[0].axisValue;
        params.forEach(function (p) {
          var v = p.seriesName === 'top1 中位'
            ? Number(p.value).toFixed(3)
            : Number(p.value).toFixed(0) + '%';
          out += '<br/>' + p.marker + p.seriesName + '：' + v;
        });
        return out;
      }
    },
    legend: {
      data: ['top1 中位', '≥0.6 占比'],
      textStyle: { color: ink, fontSize: 12 },
      top: 0
    },
    grid: { left: 56, right: 56, top: 52, bottom: 62 },
    xAxis: {
      type: 'category',
      data: ['中文查询\n319 次', '选择题\n2,218 次', '英文问答\n4,580 次', '统计聚合\n867 次', 'claim 类\n254 次（修复前）'],
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: { color: ink, fontSize: 11, lineHeight: 15 }
    },
    yAxis: [
      {
        type: 'value',
        name: 'top1 中位分数',
        min: 0,
        max: 1,
        nameTextStyle: { color: muted, fontSize: 11 },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: muted, fontSize: 11 },
        splitLine: { lineStyle: { color: rule, type: 'dashed' } }
      },
      {
        type: 'value',
        name: '≥0.6 占比',
        min: 0,
        max: 100,
        nameTextStyle: { color: muted, fontSize: 11 },
        axisLine: { show: false },
        axisTick: { show: false },
        axisLabel: { color: muted, fontSize: 11, formatter: '{value}%' },
        splitLine: { show: false }
      }
    ],
    series: [
      {
        name: 'top1 中位',
        type: 'bar',
        barWidth: '30%',
        yAxisIndex: 0,
        data: [0.912, 0.904, 0.859, 0.811, 0.300],
        itemStyle: {
          color: function (p) {
            return p.dataIndex === 4 ? accent2 : accent;
          }
        },
        label: {
          show: true, position: 'top', fontSize: 11, fontWeight: 700,
          color: ink, formatter: function (p) { return Number(p.value).toFixed(3); }
        }
      },
      {
        name: '≥0.6 占比',
        type: 'line',
        yAxisIndex: 1,
        data: [88, 78, 74, 66, 15],
        symbol: 'circle',
        symbolSize: 7,
        lineStyle: { width: 2, color: accent2, type: 'dashed' },
        itemStyle: { color: accent2 },
        label: {
          show: true, position: 'bottom', color: accent2, fontSize: 11,
          formatter: '{c}%'
        }
      }
    ]
  });

  window.addEventListener('resize', function () { chart.resize(); });
})();
