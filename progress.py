from progressbar import ProgressBar

def create_progressbar(max_value):
    return ProgressBar(max_value=max_value, widget_kwargs={
        'marker': '\u2588',
        'fill': ':'
    })

def finish_progressbar(progbar):
    progbar.finish()
    print()
