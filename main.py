import logging
import json
import telegram
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from mlutils import MLProcessor
import numpy as np
import os
from io import BytesIO
from PIL import Image
from utils import blurr_mask, infer_mask


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

with open('config.json') as file:
    mlprocessor = MLProcessor(json.load(file))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "I am inpainting bot. I use diffusion model + DisenBooth to inpaint your images.\n"
    text = text + "Please send me one with prompt in the caption.\nI support the following concepts:\n"

    concepts = mlprocessor.list_concepts()
    for name, desc in concepts.items():
        text = text + '\t\t - ' + name + ' - ' + desc + '\n'

    await update.message.reply_text(
        text
    )


async def change_negative_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['negative_prompt'] = update.message.text
    await update.message.reply_text('Negative prompt set to: ' + update.message.text)


async def inpaint(update: Update, context: ContextTypes.DEFAULT_TYPE):
    negative_prompt = context.user_data.get('negative_prompt', None)
    prompt = update.message.caption
    if prompt is None:
        prompt = ""

    file = await context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(await file.download_as_bytearray())
    init_image = Image.open(f)
    max_side = max(init_image.size[0], init_image.size[1])
    if max_side > 512:
        w = int(init_image.size[0] * 512 / max_side)
        h = int(init_image.size[1] * 512 / max_side)
        init_image = init_image.resize((w, h), Image.Resampling.LANCZOS)

    mask = infer_mask(init_image)
    mask = blurr_mask(mask)
    images = mlprocessor.generate(init_image, mask_image=mask, 
                                  prompt=prompt, negative_prompt=negative_prompt)
    tg_images = []
    for image in images:
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        tg_images.append(telegram.InputMediaPhoto(img_byte_arr))
    
    await update.message.reply_media_group(tg_images) 
    #caption=f"Prompt: {prompt}\nNegative prompt: {negative_prompt}")


def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(os.environ["TOKEN"]).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(CommandHandler("negative", change_negative_prompt))
    application.add_handler(MessageHandler(filters.PHOTO, inpaint))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
