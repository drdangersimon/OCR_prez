import editdistance
from numpy import mean

def edit_distance(y_true, y_pred):
    mean_ed = mean([editdistance.eval(y_true[j], y_pred[j]) for j in range(len(y_true))])
    mean_norm_ed = mean_ed / max(map(len, y_pred))
    return mean_ed, mean_norm_ed