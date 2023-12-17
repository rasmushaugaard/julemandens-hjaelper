import uuid
import pathlib
import signal
import time

import wget
from openai import OpenAI
import tqdm

client = OpenAI()

prompts = {
    #'good': "A top-down photograph of a christmas present in Santa's workshop. The present is centered in the image.",
    #'good_belt': "A top-down photograph of a christmas present on a conveyer belt in Santa's workshop. The present is centered in the image.",
    #'bad_destroyed': "A top-down photograph of a destroyed christmas present on a conveyer belt in Santa's workshop. The destroyed present is centered in the image.",
    #'bad_explode': "A top-down photograph of an exploded christmas present in Santa's workshop. The present is centered in the image.",
    'bad_broken': "A top-down photograph of a completely broken and destroyed christmas present in Santa's workshop. The broken present is centered in the image.",
    'bad_damaged': "A top-down photograph of a damaged christmas present in Santa's workshop. The damaged present is in the center of the image.",
    #'bad_vandalized': "A top-down photograph of a vandalized christmas present in Santa's workshop. The present is centered in the image.",
}

should_close = []
signal.signal(signal.SIGINT, lambda *_: should_close.append(1))

for folder, prompt in prompts.items():
    folder = pathlib.Path('images') / folder
    folder.mkdir(exist_ok=True)

    n = 20 - len(list(folder.glob('*.png')))
    for _ in tqdm.tqdm(range(n), total=n):
        start = time.time()

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        wget.download(response.data[0].url, str(folder / f'{uuid.uuid4()}.png'))
        
        if should_close:
            quit()
        
        duration = time.time() - start
        # rate limit = 5 img/minute
        safe_buffer = 10
        time.sleep(max(0, (60 + safe_buffer)/5 - duration))