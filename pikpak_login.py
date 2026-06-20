import asyncio
import json
import getpass

from pikpakapi import PikPakApi

import config


def save_token(client: PikPakApi):
    """把登录后的 token 等信息存到本地，供后续脚本复用。"""
    data = client.to_dict()
    with open(config.TOKEN_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n✅ token 已保存到: {config.TOKEN_PATH}")


async def main():
    print("=" * 50)
    print("PikPak 登录 —— 第一步：获取并保存 token")
    print("=" * 50)
    print(f"账号(邮箱): {config.PIKPAK_USERNAME}")

    if "你的邮箱" in config.PIKPAK_USERNAME:
        print("\n❌ 请先在 config.py 里把 PIKPAK_USERNAME 改成你的真实邮箱！")
        return

    # 密码运行时手动输入，不回显
    password = getpass.getpass("请输入 PikPak 密码（输入时不显示）: ")

    client = PikPakApi(
        username=config.PIKPAK_USERNAME,
        password=password,
        device_id=config.DEVICE_ID,   # 固定设备指纹
    )

    try:
        print("\n正在登录 ...")
        await client.login()
        print("✅ 登录成功！")

        # 验证 token 真的可用：拉一下网盘容量信息
        print("正在验证 token（获取网盘容量）...")
        quota = await client.get_quota_info()
        q = quota.get("quota", {})
        limit = int(q.get("limit", 0)) / (1024**3)
        usage = int(q.get("usage", 0)) / (1024**3)
        print(f"✅ token 有效！网盘容量: 已用 {usage:.1f} GB / 共 {limit:.1f} GB")

        # 保存 token
        save_token(client)
        print("\n🎉 第一步完成！下一步我们用这个 token 去列目录、抓缩略图。")

    except Exception as e:
        print(f"\n❌ 登录失败: {e}")
        print("\n可能原因与对策：")
        print("  1. 密码错误 —— 重新确认。")
        print("  2. 触发验证码/风控 —— 换成你平时登录 PikPak 的网络环境再试。")
        print("  3. 接口失效 —— 非官方 API 偶发，过段时间或等库更新再试。")
        print("  4. 如多次失败，我们改用『手动获取 token 填入』的备用方案。")
    finally:
        # 关闭 httpx 连接
        try:
            await client.httpx_client.aclose()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
