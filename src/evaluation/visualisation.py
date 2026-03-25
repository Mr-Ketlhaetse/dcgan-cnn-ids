import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from torch.utils.tensorboard import SummaryWriter


def plot_and_log(grid_search, scoring, image_dataset, labels, cv, plots_dir, tensorboard_dir):
    """
    Generate per-metric score plots, learning curves, and TensorBoard entries.

    Args:
        grid_search: Fitted GridSearchCV object.
        scoring (dict): Scorer name → scorer mapping used during grid search.
        image_dataset: Dataset passed to learning_curve.
        labels: Label tensor passed to learning_curve.
        cv (int): Cross-validation folds.
        plots_dir (str): Directory to save PNG plots.
        tensorboard_dir (str): Directory for TensorBoard logs.
    """
    os.makedirs(plots_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    for scorer_name in scoring.keys():
        mean_scores = grid_search.cv_results_['mean_test_' + scorer_name]
        print(f"{scorer_name.capitalize()} Score: {mean_scores}")

        # Score vs parameter combination
        plt.plot(mean_scores)
        plt.xlabel('Parameter Combination')
        plt.ylabel(f"{scorer_name.capitalize()} Score")
        plt.title(f"{scorer_name.capitalize()} Score vs Parameter Combination")
        plt.ylim([0, 1])
        plt.savefig(os.path.join(plots_dir, f"{scorer_name}.png"))
        plt.clf()

        # Learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            grid_search.best_estimator_, image_dataset, labels,
            cv=cv, scoring=scorer_name, n_jobs=1
        )
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        fig = plt.figure()
        plt.title(f"Learning Curve ({scorer_name.capitalize()} Score)")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                         alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                         alpha=0.1, color="g")
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.savefig(os.path.join(plots_dir, f"learning_curve_{scorer_name}.png"))
        plt.clf()
        plt.close(fig)

        # TensorBoard
        mean_score = float(np.mean(mean_scores))
        writer.add_scalar(f"{scorer_name.capitalize()} Score", mean_score)
        writer.add_histogram(f"{scorer_name.capitalize()} Score Dist", mean_scores)

    writer.close()
