import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go


# df1 = pd.DataFrame(dict(
#     s0=[55, 55, 55, 75, 75, 75, 100, 100, 100],
#     t=[2, 3, 5, 2, 3, 5, 2, 3, 5],
#     rn50_cifar100_natural=[42.4, 43.6, 46.4,  42.8, 43.9, 46.3,  43.2, 45.5, 46.0],
#     rn50_cifar100_adv=[14.7, 13.7, 12.0,  14.6, 13.6, 12.5,  14.0, 13.7, 11.9],
#      rn18cifar100_natural=[45.3, 48.8, 47.1,  45.7, 47.3, 45.8,  47.7, 46.2, 46.9],
#      rn18cifar100_adversarial= [13.3, 12.1, 9.7,  12.7, 11.9, 10,  13, 10.8, 9.9],
# ))
#
# fig1 = go.Figure()
# fig1 = px.scatter(df1, x="s0",
#              y='rn18cifar100_adversarial',
#              title="ResNet-18 - CIFAR-100 adversarial accuracy",
#              #barmode='group',
#              height=700,
#              facet_col="t"
#             ).update_traces(mode="lines+markers")
#
# # fig2 = px.scatter(df1, x="s0", color="rn18cifar100_natural",
# #              y='rn18cifar100_natural',
# #              title="A Grouped Bar Chart With Plotly Express in Python",
# #              #barmode='group',
# #              height=700,
# #              facet_col="t"
# #             ).update_traces(mode="lines+markers")
# #
# #
# # fig3 = go.Figure(data=fig1.data + fig2.data)
#
# plotly.offline.plot(fig1, filename='interactive_plots/cifar-100.html')

# data_1 = {
#     "rn18cifar100_natural_55-2":[45.3],
#              "rn18cifar100_natural_55-3":[48.8],
#     "rn18cifar100_natural_55-5":[47.1],
#     "rn18cifar100_natural_75-2":[45.7],
#              "rn18cifar100_natural_75-3":[47.3],
#     "rn18cifar100_natural_75-5":[45.8],
#     "rn18cifar100_natural_100-2":[47.7],
#              "rn18cifar100_natural_100-3":[46.2],
#     "rn18cifar100_natural_100-5":[46.9],
#
#     "rn18cifar100_adversarial_55-2": [13.3],
#              "rn18cifar100_adversarial_55-3":[12.1],
#     "rn18cifar100_adversarial_55-5":[9.7],
#     "rn18cifar100_adversarial_75-2":[12.7],
#              "rn18cifar100_adversarial_75-3":[11.9],
#     "rn18cifar100_adversarial_75-5":[10.0],
#     "rn18cifar100_adversarial_100-2": [13.0],
#              "rn18cifar100_adversarial_100-3":[10.8]
#     "rn18cifar100_adversarial_100-5":[9.9],
#
#     "s0": [55,55,55,75,75,75,100,100,100],
#
#
#     "t": [2,3,5,2,3,5,2,3,5],
#
# }
#
# fig1 = go.Figure(
#     data = [
#         go.Scatter(
#             x=data_1["t"],
#             y=data_1["rn18cifar100_natural_55"],
#             #offsetgroup=0,
#         ),
#         go.Scatter(
#             x=data_1["t"],
#             y=data_1["rn18cifar100_natural_75"],
#             #offsetgroup=0,
#         ),
#         go.Scatter(
#             x=data_1["t"],
#             y=data_1["rn18cifar100_natural_100"],
#             #offsetgroup=0,
#         ),
#
#         go.Scatter(
#             x=data_1["t"],
#             y=data_1["rn18cifar100_adversarial_55"],
#             # offsetgroup=0,
#         ),
#         go.Scatter(
#             x=data_1["t"],
#             y=data_1["rn18cifar100_adversarial_75"],
#             # offsetgroup=0,
#         ),
#         go.Scatter(
#             x=data_1["t"],
#             y=data_1["rn18cifar100_adversarial_100"],
#             # offsetgroup=0,
#         ),
#
#         # go.Scatter(
#         #     name="adversarial RN18",
#         #     x=data_1["labels"],
#         #     y=data_1["rn18cifar100_adversarial"],
#         #     #offsetgroup=0,
#         # ),
#     ],
#     layout=go.Layout(
#         title="Natural and Adversarial Accuracy (CIFAR-100)",
#         #yaxis_title="Number of Issues"
#         yaxis_title="Accuracy",
#         xaxis_title="s0,t"
#     )
# )
# plotly.offline.plot(fig1, filename='interactive_plots/cifar-100.html')



data_1 = {
    "rn50cifar100_natural": [42.4, 43.6, 46.4, None, 42.8, 43.9, 46.3, None, 43.2, 45.5, 46.0],

    "rn18cifar100_natural": [45.3, 48.8, 47.1, None, 45.7, 47.3, 45.8, None, 47.7, 46.2, 46.9],

    "rn50cifar10_natural": [79.6, 70.5, 66, None, 81, 70.7, 65.1, None, 80.4, 69.8, 63.3],
    "rn18cifar10_natural": [79.2, 69.7, 60.9, None, 79, 68.4, 64.4, None, 80.1, 69.7, 71.2],

    "labels": [
        "s0=55, t=2",
        "s0=55, t=3",
        "s0=55, t=5",
        None,
        "s0=75, t=2",
        "s0=75, t=3",
        "s0=75, t=5",
        None,
        "s0=100, t=2",
        "s0=100, t=3",
        "s0=100, t=5",
    ]
}

fig1 = go.Figure(
    data = [
        go.Scatter(
            name="RN50-CIFAR-100",
            x=data_1["labels"],
            y=data_1["rn50cifar100_natural"],
            #offsetgroup=0,
        ),
        go.Scatter(
            name="RN18-CIFAR-100",
            x=data_1["labels"],
            y=data_1["rn18cifar100_natural"],
            #offsetgroup=0,
        ),
        go.Scatter(
            name="RN50-CIFAR-10",
            x=data_1["labels"],
            y=data_1["rn50cifar10_natural"],
            #offsetgroup=0,
            line_color='#008000',
        ),

        go.Scatter(
            name="RN18-CIFAR-10",
            x=data_1["labels"],
            y=data_1["rn18cifar10_natural"],
            #offsetgroup=0,
            line_color='#FF8C00',

        ),
    ],
    layout=go.Layout(
        title="Natural Accuracy versus s0, t",
        #yaxis_title="Number of Issues"
        yaxis_title="Accuracy",
        xaxis_title="s0,t",
        #paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Arial, monospace",
            size=20

        ),
    legend=dict(
                yanchor="middle",
                y=0.92,
                xanchor="center",
                x=0.94
            )
    )
)

# fig = go.Figure(data=[go.Bar(
#     name = 'Data 1',
#     x = [55, 55, 55, 75, 75, 75, 100, 100, 100],
#     y = [42.4, 43.6, 46.4,  42.8, 43.9, 46.3,  43.2, 45.5, 46]
#    ),
#                        go.Bar(
#     name = 'Data 2',
#     x = [55, 55, 55, 75, 75, 75, 100, 100, 100],
#     y = [14.7, 13.7, 12,  14.6, 13.6, 12.5,  14, 13.7, 11.9]
#    )
# ])


# fig.show()
plotly.offline.plot(fig1, filename='interactive_plots/natural.html')



data_2 = {
    "rn50cifar100_adversarial": [14.7, 13.7, 12.0, None, 14.6, 13.6, 12.5, None, 14.0, 13.7, 11.9],
    "rn18cifar100_adversarial": [13.3, 12.1, 9.7, None, 12.7, 11.9, 10.0, None, 13.0, 10.8, 9.9],
    "rn50cifar10_adversarial": [30.9, 28.2, 24.3, None, 32.3, 29.1, 21.9, None, 31.8, 28.9, 21.7],
    "rn18cifar10_adversarial": [28.7, 27.4, 21.7, None, 25.6, 27.6, 20, None, 28.5, 27, 21.7],


    "labels": [
        "s0=55, t=2",
        "s0=55, t=3",
        "s0=55, t=5",
        None,
        "s0=75, t=2",
        "s0=75, t=3",
        "s0=75, t=5",
        None,
        "s0=100, t=2",
        "s0=100, t=3",
        "s0=100, t=5",
    ]
}

fig2 = go.Figure(
    data = [
        go.Scatter(
            name="RN50-CIFAR-100",
            x=data_2["labels"],
            y=data_2["rn50cifar100_adversarial"],
            #offsetgroup=0,
        ),
        go.Scatter(
            name="RN18-CIFAR-100",
            x=data_2["labels"],
            y=data_2["rn18cifar100_adversarial"],
            #offsetgroup=0,
        ),
        go.Scatter(
            name="RN50-CIFAR-10",
            x=data_2["labels"],
            y=data_2["rn50cifar10_adversarial"],
            #offsetgroup=0,
            line_color='#008000',
        ),

        go.Scatter(
            name="RN18-CIFAR-10",
            x=data_2["labels"],
            y=data_2["rn18cifar10_adversarial"],
            #offsetgroup=0,
            line_color='#FF8C00',

        ),
    ],
    layout=go.Layout(
        title="Adversarial Accuracy versus s0, t",
        #yaxis_title="Number of Issues"
        yaxis_title="Accuracy",
        xaxis_title="s0,t",
        #paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(
            family="Arial, monospace",
            size=20

        ),

        legend=dict(
                yanchor="middle",
                y=0.92,
                xanchor="center",
                x=0.94
            )
    )
)

# fig = go.Figure(data=[go.Bar(
#     name = 'Data 1',
#     x = [55, 55, 55, 75, 75, 75, 100, 100, 100],
#     y = [42.4, 43.6, 46.4,  42.8, 43.9, 46.3,  43.2, 45.5, 46]
#    ),
#                        go.Bar(
#     name = 'Data 2',
#     x = [55, 55, 55, 75, 75, 75, 100, 100, 100],
#     y = [14.7, 13.7, 12,  14.6, 13.6, 12.5,  14, 13.7, 11.9]
#    )
# ])





# fig.show()
plotly.offline.plot(fig2, filename='interactive_plots/adversarial.html')