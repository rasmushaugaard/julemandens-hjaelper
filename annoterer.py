from pathlib import Path
import ipywidgets as widgets

import utils
import config

class Annoterer:
    def __init__(self, image_folder=config.images_folder, 
                 anno_fp=config.anno_fp, debug=False):
        self.anno_fp = anno_fp
        assert image_folder.is_dir()
        image_fps = set(image_folder.glob('*.jpg'))
        assert image_fps
        self.n = len(image_fps)
        
        annotations = utils.load_annotations(anno_fp)
        image_fps_annotated = set((utils.img_fp_from_id(a[0]) for a in annotations))
        self.i = len(image_fps_annotated)
        
        image_fps_need_anno = image_fps - image_fps_annotated
        self.image_fps = list(image_fps_annotated) + list(image_fps_need_anno)
        
        self.label = widgets.Label(
            'Er julegaven ok?', 
            style=dict(font_size='1.7em'),
        )
        knap_ja = widgets.Button(description="Ja")
        knap_nej = widgets.Button(description="Nej")
        knap_ja.on_click(lambda _: self.annoter(True))
        knap_nej.on_click(lambda _: self.annoter(False))
        
        self.knapper = [knap_ja, knap_nej]
        if debug:
            knap_delete = widgets.Button(description="Slet")
            knap_delete.on_click(self.slet)
            self.knapper.append(knap_delete)
    
        self.pbar = widgets.IntProgress(
            value=self.i, min=self.i, max=self.n, 
            description=self.get_pbar_text(),
            style=dict(description_width='auto'),
            layout=dict(width='500px'),
        )

        self.image = widgets.Image(width=500)
        self.fp_label = widgets.Label() if debug else None
        self.update_image()
        self.er_færdig()

        ws = list(filter(lambda x: x, [
            widgets.HBox([self.label], layout=dict(justify_content='center')),
            self.image,
            self.fp_label,
            self.pbar,
            widgets.HBox(self.knapper),
        ]))
        
        self.widget = widgets.Box(ws, layout=dict(
            display='flex',
            flex_flow='column',
            align_items='center', 
            width='500px',
        ))
        
    def update_image(self):
        fp = self.image_fps[min(self.n - 1, self.i)]
        with fp.open('rb') as f:
            self.image.value = f.read()
        if self.fp_label:
            self.fp_label.value = str(fp)
    
    def slet(self, _):
        if self.er_færdig():
            return
        img_id = utils.img_id_from_fp(self.image_fps[self.i])
        fp = utils.original_img_fp_from_id(img_id)
        assert fp.exists()
        fp.unlink()
        self.næste()
    
    def get_pbar_text(self):
        return f'{self.i} / {self.n} billeder annoteret:'
    
    def færdig(self):
        self.label.value = "Godt arbejde!"
        for knap in self.knapper:
            knap.disabled = True
            
    def er_færdig(self):
        if self.i >= self.n:
            self.færdig()
            return True
        return False
    
    def næste(self):
        self.i += 1
        self.pbar.value = self.i
        self.pbar.description = self.get_pbar_text()
        if self.i == self.n:
            self.færdig()
        else:
            self.update_image()
    
    def annoter(self, ok: bool):
        if self.er_færdig():
            return
        img_id = utils.img_id_from_fp(self.image_fps[self.i])
        with self.anno_fp.open('a') as f:
            f.write(f'{img_id} {ok}\n')
        self.næste()