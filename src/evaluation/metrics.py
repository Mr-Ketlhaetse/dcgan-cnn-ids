from sklearn.metrics import confusion_matrix


def false_alarm_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 1:
        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    return fp / (fp + tn) if fp + tn > 0 else 0


def false_negative_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 1:
        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    return fn / (fn + tp) if fn + tp > 0 else 0


def true_negative_rate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 1:
        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if tn + fp > 0 else 0
