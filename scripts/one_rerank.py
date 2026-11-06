import asyncio
import httpx
import bench_rerank as b


async def main():
    p = b.make_prompt("我上次胃镜检查结果是什么，医生让我什么时候复查？", 8, 300)
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(
            b.URL,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + b.API_KEY},
            json={
                "model": b.MODEL,
                "messages": [
                    {"role": "system", "content": b.SYS},
                    {"role": "user", "content": p},
                ],
                "temperature": 0.0,
                "max_tokens": 768,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        d = r.json()
        print(d["choices"][0]["message"]["content"])
        print("USAGE", d.get("usage"))


asyncio.run(main())
