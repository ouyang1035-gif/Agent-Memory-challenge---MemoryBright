# -*- coding: utf-8 -*-
"""本地评测集：覆盖 AML 文本赛道的核心能力维度。

每个样本：先 Add 一段多轮对话，再用若干 question 做 Search，
每个 question 标注「必须召回的关键证据关键词」和「不应召回的干扰词」。
评估检索质量（Precision/Recall/证据命中），而非最终答案（答案由平台生成）。
"""

# 每个 case: {"user_id", "conversations": [...], "questions": [...]}
# conversations: 模拟平台分块发来的 messages
# questions: {"query", "must_have": [关键词...], "must_not": [...], "options": [...], "dimension": "..."}

EVAL_CASES = [
    # ===== 维度 A：显式事实召回 =====
    {
        "user_id": "eval-fact",
        "dimension": "explicit_fact",
        "conversations": [
            [
                {"role": "user", "content": "我在杭州的一家互联网公司做产品经理", "timestamp": 1704067200000},
                {"role": "assistant", "content": "产品经理的工作节奏怎么样", "timestamp": 1704067210000},
                {"role": "user", "content": "挺忙的，我负责的是支付这条业务线", "timestamp": 1704067220000},
            ]
        ],
        "questions": [
            {
                "query": "用户负责哪条业务线？",
                "must_have": ["支付"],
                "must_not": [],
            },
            {
                "query": "用户在哪个城市工作？",
                "must_have": ["杭州"],
                "must_not": [],
            },
        ],
    },
    # ===== 维度 B：多跳关系 =====
    {
        "user_id": "eval-multihop",
        "dimension": "multi_hop",
        "conversations": [
            [
                {"role": "user", "content": "我周末经常去西湖边跑步", "timestamp": 1704067200000},
                {"role": "assistant", "content": "西湖边跑步很舒服", "timestamp": 1704067210000},
            ],
            [
                {"role": "user", "content": "我加入了一个跑团，团长叫大鹏", "timestamp": 1704153600000},
                {"role": "assistant", "content": "跑团氛围好吗", "timestamp": 1704153610000},
                {"role": "user", "content": "大鹏组织了每周三的间歇训练", "timestamp": 1704153620000},
            ],
        ],
        "questions": [
            {
                "query": "用户参加的那个跑团的组织者安排了什么训练？",
                "must_have": ["大鹏", "间歇训练"],
                "must_not": [],
            },
        ],
    },
    # ===== 维度 C：时间与事件序列 =====
    {
        "user_id": "eval-temporal",
        "dimension": "temporal",
        "conversations": [
            [
                {"role": "user", "content": "我 2023 年 3 月入职了现在的公司", "timestamp": 1704067200000},
            ],
            [
                {"role": "user", "content": "2023 年 6 月我转正了，加薪了 15%", "timestamp": 1704153600000},
            ],
            [
                {"role": "user", "content": "2024 年 1 月我升职成高级产品经理", "timestamp": 1704240000000},
            ],
        ],
        "questions": [
            {
                "query": "用户什么时候转正的？",
                "must_have": ["2023", "6 月"],
                "must_not": [],
            },
            {
                "query": "用户入职后先转正还是先升职？",
                "must_have": ["转正", "2023"],
                "must_not": [],
            },
        ],
    },
    # ===== 维度 D：记忆治理（冲突消解/纠正）=====
    {
        "user_id": "eval-governance",
        "dimension": "governance",
        "conversations": [
            [
                {"role": "user", "content": "我住在滨江区，每天骑车上班", "timestamp": 1704067200000},
            ],
            [
                {"role": "user", "content": "不对，我上个月搬家了，现在住在余杭区，改坐地铁了", "timestamp": 1704672000000},
            ],
        ],
        "questions": [
            {
                "query": "用户现在住在哪个区？",
                "must_have": ["余杭"],
                "must_not": [],  # 滨江可能被召回但应被 superseded 过滤或排序靠后
            },
        ],
    },
    # ===== 维度 E：个性化偏好 =====
    {
        "user_id": "eval-preference",
        "dimension": "preference",
        "conversations": [
            [
                {"role": "user", "content": "我特别喜欢悬疑电影，比如《看不见的客人》", "timestamp": 1704067200000},
                {"role": "assistant", "content": "那部反转很多", "timestamp": 1704067210000},
                {"role": "user", "content": "对，但我不太喜欢恐怖片，太吓人了", "timestamp": 1704067220000},
            ]
        ],
        "questions": [
            {
                "query": "给我推荐一部电影",
                "must_have": ["悬疑"],
                "must_not": [],
                "dimension": "preference",
            },
        ],
    },
    # ===== 维度：列表题（须覆盖全部事实）=====
    {
        "user_id": "eval-list",
        "dimension": "list_recall",
        "conversations": [
            [
                {"role": "user", "content": "我会 Python、Go 和 Rust 三门语言", "timestamp": 1704067200000},
            ]
        ],
        "questions": [
            {
                "query": "用户会哪些编程语言？",
                "must_have": ["Python", "Go", "Rust"],
                "must_not": [],
            },
        ],
    },
    # ===== 维度：MCQ 选择题 =====
    {
        "user_id": "eval-mcq",
        "dimension": "mcq",
        "conversations": [
            [
                {"role": "user", "content": "我家养了一只金毛，叫豆豆，今年三岁了", "timestamp": 1704067200000},
            ]
        ],
        "questions": [
            {
                "query": "用户的宠物狗叫什么名字？",
                "options": ["A. 球球", "B. 豆豆", "C. 旺财", "D. 乐乐"],
                "must_have": ["豆豆"],
                "must_not": [],
            },
        ],
    },
    # ===== 维度：噪声抵抗（无关记忆不应排在前面）=====
    {
        "user_id": "eval-noise",
        "dimension": "noise_resistance",
        "conversations": [
            [
                {"role": "user", "content": "我的生日是 5 月 20 日", "timestamp": 1704067200000},
                {"role": "assistant", "content": "记下了", "timestamp": 1704067210000},
                {"role": "user", "content": "今天天气真好", "timestamp": 1704067220000},
                {"role": "assistant", "content": "是啊适合出门", "timestamp": 1704067230000},
                {"role": "user", "content": "中午吃了麻辣烫", "timestamp": 1704067240000},
            ]
        ],
        "questions": [
            {
                "query": "用户的生日是哪天？",
                "must_have": ["5 月 20"],
                "must_not_in_top1": ["麻辣烫", "天气"],  # 噪声不应排第一
            },
        ],
    },
]
