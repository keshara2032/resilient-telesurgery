import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import altair as alt
from altair_saver import save
import altair_viewer
import matplotlib.pyplot as plt


def get_classification_report(pred, gt, target_names):
    report = classification_report(gt, pred, target_names=target_names, output_dict=True)
    return pd.DataFrame(report).transpose()

def visualize_gesture_ts(pred, gt, target_names):

    def _convert_label_to_range(labels):
        df = pd.DataFrame({'gesture': labels})
        pred_index_changes = df["gesture"].diff()[df["gesture"].diff() != 0].index.values

        changes_df = df.iloc[pred_index_changes]
        changes_df.reset_index(inplace=True)
        # changes_df.drop(columns=changes_df.columns[0], axis=1, inplace=True)

        index_change = []
        for idx, _ in changes_df.iterrows():
            if(idx < changes_df.shape[0]-1):
                index_change.append([changes_df.iloc[idx]["index"], changes_df.iloc[idx+1]["index"], changes_df.iloc[idx]["gesture"]])

        gesture_range_df  = pd.DataFrame(index_change)
        gesture_range_df.columns = ["start","end","gesture"]

        label_mappings = {i: target_names[i] for i in range(len(target_names))}
        gesture_range_df["gesture"] = gesture_range_df["gesture"].map(label_mappings)

        return gesture_range_df
    
    pred_gesture_ranges_df = _convert_label_to_range(pred)
    gt_gesture_ranges_df = _convert_label_to_range(gt)

    # pred = alt.Chart(pred_gesture_ranges_df).mark_bar(clip=True).encode(
    #     x=alt.X('start', scale=alt.Scale(domain=[0,3000])),
    #     x2='end',
    #     y=alt.Y('sum(gesture)',title = "Gesture", axis=alt.Axis(labels=False)),
    #     color=alt.Color('gesture', scale=alt.Scale(scheme='dark2'))
    # ).properties(
    #     width=800,
    #     height=25,
    #     title="Prediction"
    # )

    # gt = alt.Chart(gt_gesture_ranges_df).mark_bar(clip=True).encode(
    #     x=alt.X('start', scale=alt.Scale(domain=[0,3000])),
    #     x2='end',
    #     y=alt.Y('sum(gesture)',title = "Gesture", axis=alt.Axis(labels=False)),
    #     color=alt.Color('gesture', scale=alt.Scale(scheme='dark2'))
    # ).properties(
    #     width=800,
    #     height=25,
    #     title="Ground Truth"
    # )

    # alt.vconcat(
    # gt.mark_bar(clip=True),
    # pred.mark_bar(clip=True),
    # )

    plt.scatter(np.arange(pred.shape[0]), pred, c='red')
    plt.scatter(np.arange(pred.shape[0]), gt, c='blue')
    plt.show()