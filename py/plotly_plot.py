from toolz import compose
from plotly.graph_objs import *
from plot_data import average, lowpass_filter
import data_handling as handler


def generate_tiled_plot(filelabel, tile=True, factor=50):
    plot_data = []
    t = None
    func = compose(average, lowpass_filter)
    for i, (l, f) in enumerate(filelabel):
        offset = i * factor if tile else 0
        data = handler.extract_data(f)
        if type(t) == type(None):
            t = data['t']
        y, fs = func((data['stim'], data['fs']))
        plot_data += [Scatter(x=t*1e3, y=y[:,0]*1e6 + offset, name=str(l), line=dict(width=2))]
    fig = {'data': plot_data}
    fig['layout'] = dict(
        title='Current-response comparison',
        xaxis=dict(title='Time (ms)'),
        yaxis=dict(title='Current increasing -->')
    )

    return fig
