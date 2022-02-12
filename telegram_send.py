
import requests


def send_images(images_path):

    chat_id = "udrmmm"
    bot_id = "5237047300:AAGA07zUAuGWEMuiycuyIJHNnV5rWyAn2Ko"
    try:
        pict = open(images_path,"rb")
        img_json = {
            "photo":pict
                    }

        send = requests.post(f"https://api.telegram.org/bot{bot_id}/sendPhoto?chat_id={chat_id}",
                      files=img_json)

        if send.status_code > 299:
            print('TELEGRAM_ALARM_PUSH_ERROR: ', send.text)

    except Exception as exc:
        print('telegram push error: ' + str(exc))

send_images("./deneme.png")