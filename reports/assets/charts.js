(function () {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();

  var axisCommon = {
    axisLine: { lineStyle: { color: rule } },
    axisTick: { show: false },
    axisLabel: { color: muted, fontSize: 11 },
    splitLine: { lineStyle: { color: rule, type: 'dashed' } }
  };

  // --- Chart: 本地回归集 P0-b 前后 ---
  var chartLocal = echarts.init(document.getElementById('chart-local'), null, { renderer: 'svg' });
  chartLocal.setOption({
    animation: false,
    color: [muted, accent],
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      valueFormatter: function (v) { return Number(v).toFixed(1) + '%'; }
    },
    legend: {
      data: ['改造前（P0-a）', '改造后（P0-b）'],
      textStyle: { color: ink, fontSize: 12 },
      top: 0
    },
    grid: { left: 64, right: 24, top: 66, bottom: 34 },
    xAxis: {
      type: 'category',
      data: ['hit@1', 'MRR（×100）', 'coverage@10'],
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: { color: ink, fontSize: 12 }
    },
    yAxis: {
      type: 'value',
      max: 100,
      name: '得分（%）',
      nameTextStyle: { color: muted, fontSize: 11 },
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: axisCommon.axisLabel,
      splitLine: axisCommon.splitLine
    },
    series: [
      {
        name: '改造前（P0-a）',
        type: 'bar',
        barWidth: '26%',
        data: [57.9, 66.3, 37.5],
        itemStyle: { color: muted },
        label: { show: true, position: 'top', color: muted, fontSize: 11, formatter: '{c}%' }
      },
      {
        name: '改造后（P0-b）',
        type: 'bar',
        barWidth: '26%',
        data: [73.7, 78.6, 100.0],
        itemStyle: { color: accent },
        label: { show: true, position: 'top', color: accent, fontSize: 11, fontWeight: 700, formatter: '{c}%' }
      }
    ]
  });
  window.addEventListener('resize', function () { chartLocal.resize(); });

  // --- Chart: 规模放大 ---
  var chartScale = echarts.init(document.getElementById('chart-scale'), null, { renderer: 'svg' });
  chartScale.setOption({
    animation: false,
    color: [accent, accent2],
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      valueFormatter: function (v) { return v === null || v === undefined ? '—' : Number(v).toFixed(1) + '%'; }
    },
    legend: {
      data: ['hit@1', 'retention@10'],
      textStyle: { color: ink, fontSize: 12 },
      top: 0
    },
    grid: { left: 52, right: 24, top: 66, bottom: 40 },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: ['s0 原始库', 's1 轻度', 's2 中度', 's3 重度'],
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: { color: ink, fontSize: 12 }
    },
    yAxis: {
      type: 'value',
      min: 30,
      max: 90,
      name: '得分（%）',
      nameTextStyle: { color: muted, fontSize: 11 },
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: axisCommon.axisLabel,
      splitLine: axisCommon.splitLine
    },
    series: [
      {
        name: 'hit@1',
        type: 'line',
        data: [
          { value: 73.7, label: { position: 'right', offset: [4, -12] } },
          63.2, 57.9, 52.6
        ],
        symbol: 'circle',
        symbolSize: 7,
        lineStyle: { width: 2.5, color: accent },
        itemStyle: { color: accent },
        label: { show: true, position: 'top', color: accent, fontSize: 11, formatter: '{c}%' }
      },
      {
        name: 'retention@10',
        type: 'line',
        data: [null, 57.4, 47.9, 35.8],
        symbol: 'circle',
        symbolSize: 7,
        lineStyle: { width: 2.5, color: accent2, type: 'dashed' },
        itemStyle: { color: accent2 },
        label: { show: true, position: 'bottom', color: accent2, fontSize: 11, formatter: '{c}%' }
      }
    ]
  });
  window.addEventListener('resize', function () { chartScale.resize(); });

  // --- Chart: 三方对比 mc10 200 题 ---
  var chartCompare = echarts.init(document.getElementById('chart-compare'), null, { renderer: 'svg' });
  chartCompare.setOption({
    animation: false,
    color: [accent, accent2, muted],
    tooltip: {
      trigger: 'axis',
      appendToBody: true,
      valueFormatter: function (v) { return Number(v).toFixed(1) + '%'; }
    },
    legend: {
      data: ['自研', '腾讯云向量数据库', 'mem0'],
      textStyle: { color: ink, fontSize: 12 },
      top: 0
    },
    grid: { left: 52, right: 24, top: 48, bottom: 34 },
    xAxis: {
      type: 'category',
      data: ['aic（答案在上下文中）', 'kw（关键词覆盖）', 'top1（首条命中）'],
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: { color: ink, fontSize: 12 }
    },
    yAxis: {
      type: 'value',
      max: 70,
      name: '百分比（%）',
      nameTextStyle: { color: muted, fontSize: 11 },
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: axisCommon.axisLabel,
      splitLine: axisCommon.splitLine
    },
    series: [
      {
        name: '自研',
        type: 'bar',
        barWidth: '20%',
        data: [53.3, 64.2, 62.3],
        itemStyle: { color: accent },
        label: { show: true, position: 'top', color: accent, fontSize: 11, fontWeight: 700, formatter: '{c}%' }
      },
      {
        name: '腾讯云向量数据库',
        type: 'bar',
        barWidth: '20%',
        data: [31.2, 41.1, 41.7],
        itemStyle: { color: accent2 },
        label: { show: true, position: 'top', color: muted, fontSize: 11, formatter: '{c}%' }
      },
      {
        name: 'mem0',
        type: 'bar',
        barWidth: '20%',
        data: [31.2, 45.1, 39.2],
        itemStyle: { color: muted },
        label: { show: true, position: 'top', color: muted, fontSize: 11, formatter: '{c}%' }
      }
    ]
  });
  window.addEventListener('resize', function () { chartCompare.resize(); });

  // --- Mermaid ---
  if (window.mermaid) {
    mermaid.initialize({
      startOnLoad: true,
      theme: 'neutral',
      securityLevel: 'loose',
      themeVariables: { fontSize: '14px' },
      flowchart: { htmlLabels: true, nodeSpacing: 30, rankSpacing: 42 }
    });
  }
})();
