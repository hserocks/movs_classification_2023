import asyncio
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from config_reader import config
import random


# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token=config.bot_token.get_secret_value())
# Диспетчер
dp = Dispatcher()

# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(f'Привет! Я определяю зверушку по фотографии. Отправь мне фотографию и я скажу кто на ней изображеню. \nНапиши /help и я покажу пример как я работаю')

# Хэндлер на команду /help
@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer("Вот пример того как cо мной взаимодейстовавать...")
    # тут должен быть скриншот уже готового бота (последний шаг)
    photo = "https://disk.yandex.ru/i/QnrlspsH0SbQdw"
    await bot.send_photo(message.chat.id, photo)
    await message.answer('Если хочешь протестировать, вот команда /example, отправлю тебе фотографию, которую ты можешь использовать для моего тестирования.')

# Хэндлер на команду /example
@dp.message(Command("example"))
async def cmd_example(message: types.Message):
    await message.answer("Вот случайная картинка из моей библиотеки")
    choise = random.randint(1, 10)

    if choise == 1:
        photo = "https://disk.yandex.ru/i/vGlZMYGgWp-gZA"
        await bot.send_photo(message.chat.id, photo)

    if choise == 2:
        photo = "https://disk.yandex.ru/i/7C1jrAI31CmuDw"
        await bot.send_photo(message.chat.id, photo)

    if choise == 3:
        photo = "https://disk.yandex.ru/i/PAv3uuRE4yxxmQ"
        await bot.send_photo(message.chat.id, photo)

    if choise == 4:
        photo = "https://disk.yandex.ru/i/Uso5wvKxcNCLWw"
        await bot.send_photo(message.chat.id, photo)

    if choise == 5:
        photo = "https://disk.yandex.ru/i/3P92ykGsPfjK8Q"
        await bot.send_photo(message.chat.id, photo)

    if choise == 6:
        photo = "https://disk.yandex.ru/i/O6E0ob1AAwfj-A"
        await bot.send_photo(message.chat.id, photo)

    if choise == 7:
        photo = "https://disk.yandex.ru/i/YN_c5h2jWn2aPQ"
        await bot.send_photo(message.chat.id, photo)

    if choise == 8:
        photo = "https://disk.yandex.ru/i/vMQcSSp1CDHPHw"
        await bot.send_photo(message.chat.id, photo)

    if choise == 9:
        photo = "https://disk.yandex.ru/i/UwkPoqqEByiG-w"
        await bot.send_photo(message.chat.id, photo)

    if choise == 10:
        photo = "https://disk.yandex.ru/i/r5YGjo8gdNZBOg"
        await bot.send_photo(message.chat.id, photo)

# тестирование  Callback кнопки для обратной связи предсказания от пользователя
@dp.message(Command("test"))
async def cmd_test(message: types.Message):

    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(
        text="Я парв",
        callback_data="true_answer")
    )
    builder.add(types.InlineKeyboardButton(
        text="Я не парв",
        callback_data="false_answer")
    )
    await message.answer(
        "Я думаю что это кошка...",
        reply_markup=builder.as_markup()
    )

# Приём ответа 1 от Callback кнопки 
@dp.callback_query(F.data == "true_answer")
async def true_answer(callback: types.CallbackQuery):
    await callback.message.answer("Ура! Я так и знал.")

# Приём ответа 1 от Callback кнопки
@dp.callback_query(F.data == "false_answer")
async def false_answer(callback: types.CallbackQuery):
    await callback.message.answer("Дай ка посмотрю ещё раз на фотографию.")

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())