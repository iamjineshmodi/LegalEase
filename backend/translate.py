import asyncio
from googletrans import Translator

async def main():
	t = Translator()
	ans = await t.translate("hi om how are you", src='en', dest='hi')
	print(ans.text)

asyncio.run(main())
