from controller import ComicTranslate

ct = ComicTranslate()
images = ct.load_initial_image("./test-data")
ct.on_initial_image_loaded(images)
ct.start_batch_process()