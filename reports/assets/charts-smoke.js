(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();

  var el = document.getElementById('chart-smoke');
  if (!el || !window.echarts) return;

  var chart = echarts.init(el, null, { renderer: 'svg' });
  chart.setOption({
    animation: false,
    grid: { left: 70, right: 40, top: 56, bottom: 64 },
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      formatter: function (params) {
        var out = params[0].axisValue;
        params.forEach(function (p) {
          if (p.value === null || p.value === undefined) return;
          out += '<br/>' + p.marker + p.seriesName + '：' + Number(p.value).toFixed(2);
        });
        return out;
      }
    },
    legend: {
      data: ['有效 smoke', '作废记录（参考）'],
      textStyle: { color: ink, fontSize: 12 },
      top: 0
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: ['v3 初版\n59.21', 'v4 基线\n10-29 06:18', 'P2 截断\n10-29 下午', '35B 修复\n10-30 01:20', '27B 丢 embed\n10-31 07:00', '27B 修复\n10-31 08:09'],
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: { color: ink, fontSize: 11, lineHeight: 15 }
    },
    yAxis: {
      type: 'value',
      min: 50,
      max: 80,
      name: '官方 smoke 总分',
      nameTextStyle: { color: muted, fontSize: 11 },
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [
      {
        name: '有效 smoke',
        type: 'line',
        data: [{ value: 59.21, label: { position: 'right' } }, 74.77, 66.26, 64.46, null, 65.43],
        symbol: 'circle',
        symbolSize: 8,
        lineStyle: { width: 2.5, color: accent },
        itemStyle: { color: accent },
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(26,79,138,0.14)' },
            { offset: 1, color: 'rgba(26,79,138,0.01)' }
          ])
        },
        label: {
          show: true,
          position: 'top',
          color: accent,
          fontSize: 11,
          fontWeight: 700,
          formatter: function (p) {
            return p.value === null || p.value === undefined ? '' : Number(p.value).toFixed(2);
          }
        },
        markLine: {
          silent: true,
          symbol: 'none',
          lineStyle: { color: accent2, type: 'dashed', width: 1.5 },
          label: {
            formatter: '冻结基线 74.77',
            color: accent2,
            fontSize: 11,
            position: 'insideStartTop'
          },
          data: [{ yAxis: 74.77 }]
        }
      },
      {
        name: '作废记录（参考）',
        type: 'scatter',
        data: [null, null, null, null, 64.78, null],
        symbol: 'emptyCircle',
        symbolSize: 9,
        itemStyle: { color: muted, borderWidth: 1.5 },
        label: {
          show: true,
          position: 'right',
          color: muted,
          fontSize: 11,
          formatter: function (p) { return Number(p.value).toFixed(2) + '（作废）'; }
        }
      },
      {
        name: '冻结基线',
        type: 'line',
        data: [],
        lineStyle: { color: accent2, type: 'dashed' },
        itemStyle: { color: accent2 },
        silent: true
      }
    ]
  });

  window.addEventListener('resize', function () { chart.resize(); });
})();
