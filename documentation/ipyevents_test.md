
# Minimal test of the ipyevents package (not used anymore)

Start jupyter lab

```
jupyter lab
```

Enable jupyterlab extension. You will need to click on the puzzle image on the far left of the window and click enable. 

Create a notebook and run the following.

```
from ipywidgets import Label, HTML, HBox, Image, VBox, Box, HBox
from ipyevents import Event 
from IPython.display import display
```

```
l = Label('Click or type on me!')
l.layout.border = '2px solid red'

h = HTML('Event info')
d = Event(source=l, watched_events=['click', 'keydown', 'mouseenter', 'touchmove'])

def handle_event(event):
    lines = ['{}: {}'.format(k, v) for k, v in event.items()]
    content = " ".join(lines)
    h.value = content

d.on_dom_event(handle_event)
                            
display(l, h)

```

You should see some information when you click on the red button.
