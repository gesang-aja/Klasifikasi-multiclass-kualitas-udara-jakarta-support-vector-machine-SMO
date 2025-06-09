import numpy as np
from itertools import combinations

np.random.seed(42)

class SVM_SMO:
    def __init__(self, C=1.0, tol=1e-3, max_passes=5, kernel='linear', degree=3, coef0=1):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.kernel_type = kernel
        self.degree = degree
        self.coef0 = coef0

    def kernel(self, X1, X2):
        if self.kernel_type == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel_type == 'poly':
            return (np.dot(X1, X2.T) + self.coef0) ** self.degree
        else:
            raise ValueError("Kernel tidak tersedia")

    def compute_kernel_matrix(self, X):
        return self.kernel(X, X)

    def decision_function(self, i):
        return np.dot(self.alpha * self.y, self.K[:, i]) + self.b

    def error(self, i):
        return self.decision_function(i) - self.y[i]

    def take_step(self, i1, i2):
        if i1 == i2:
            return 0
        alpha1 = self.alpha[i1]
        alpha2 = self.alpha[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]
        s = y1 * y2

        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha2 + alpha1 - self.C)
            H = min(self.C, alpha2 + alpha1)
        if L == H:
            return 0

        k11 = self.K[i1, i1]
        k12 = self.K[i1, i2]
        k22 = self.K[i2, i2]
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            f1 = y1 * (E1 + self.b) - alpha1 * k11 - s * alpha2 * k12
            f2 = y2 * (E2 + self.b) - alpha2 * k22 - s * alpha1 * k12
            Lobj = (alpha1 + s * (alpha2 - L)) * f1 + L * f2 + 0.5 * (alpha1 + s * (alpha2 - L)) ** 2 * k11 + 0.5 * L ** 2 * k22 + s * (alpha1 + s * (alpha2 - L)) * L * k12
            Hobj = (alpha1 + s * (alpha2 - H)) * f1 + H * f2 + 0.5 * (alpha1 + s * (alpha2 - H)) ** 2 * k11 + 0.5 * H ** 2 * k22 + s * (alpha1 + s * (alpha2 - H)) * H * k12
            if Lobj < Hobj - 1e-3:
                a2 = L
            elif Lobj > Hobj + 1e-3:
                a2 = H
            else:
                a2 = alpha2

        if abs(a2 - alpha2) < self.tol * (a2 + alpha2 + self.tol):
            return 0
        a1 = alpha1 + s * (alpha2 - a2)

        #update treshold
        b1 = self.b - E1 - y1 * (a1 - alpha1) * k11 - y2 * (a2 - alpha2) * k12
        b2 = self.b - E2 - y1 * (a1 - alpha1) * k12 - y2 * (a2 - alpha2) * k22
        if 0 < a1 < self.C:
            self.b = b1
        elif 0 < a2 < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        #update eror cache
        self.alpha[i1] = a1
        self.alpha[i2] = a2
        self.errors[i1] = self.error(i1)
        self.errors[i2] = self.error(i2)

        return 1

    def examine_example(self, i2):
        y2 = self.y[i2]
        alpha2 = self.alpha[i2]
        E2 = self.errors[i2]
        r2 = E2 * y2

        if (r2 < -self.tol and alpha2 < self.C) or (r2 > self.tol and alpha2 > 0):
            non_bound = np.where((self.alpha != 0) & (self.alpha != self.C))[0]
            if len(non_bound) > 1:
                i1 = np.argmax(np.abs(self.errors - E2))
                if self.take_step(i1, i2):
                  return 1
            for i1 in np.random.permutation(non_bound):
                if self.take_step(i1, i2):
                   return 1
            for i1 in np.random.permutation(len(self.alpha)):
                if self.take_step(i1, i2):
                  return 1
        return 0

    def fit(self, X, y):
        self.X = X
        self.y = y.astype(float)
        self.alpha = np.zeros(len(y))
        self.b = 0
        self.K = self.compute_kernel_matrix(X)
        self.errors = -self.y.copy()
        passes = 0
        examine_all = True

        while passes < self.max_passes:
            num_changed = 0
            if examine_all:
                for i in range(len(self.y)):
                    num_changed += self.examine_example(i)
            else:
                non_bound = np.where((self.alpha != 0) & (self.alpha != self.C))[0]
                for i in non_bound:
                    num_changed += self.examine_example(i)
            print(f"Pass {passes+1}, number changed: {num_changed}")
            if num_changed == 0:
              passes = passes + 1
              examine_all = not examine_all
            else:
              passes = 0

    def predict(self, X_test):
        K_test = self.kernel(X_test, self.X)
        return np.sign(np.dot((self.alpha * self.y), K_test.T) + self.b)



class OneVsOneSVM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.models = []
        self.class_pairs = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls1, cls2 in combinations(self.classes, 2):
            idx = np.where((y == cls1) | (y == cls2))[0]
            X_pair = X[idx]
            y_pair = y[idx]
            y_binary = np.where(y_pair == cls1, 1, -1)

            print(f"Training {cls1} vs {cls2}")
            model = SVM_SMO(**self.kwargs)
            model.fit(X_pair, y_binary)

            self.models.append((model, cls1, cls2))
            self.class_pairs.append((cls1, cls2))

    def predict(self, X):
        votes = np.zeros((X.shape[0], len(self.classes)))

        for model, cls1, cls2 in self.models:
            pred = model.predict(X)
            for i, p in enumerate(pred):
                if p == 1:
                    votes[i, cls1] += 1
                else:
                    votes[i, cls2] += 1

        return np.argmax(votes, axis=1)
