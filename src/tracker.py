import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def compute_iou(bb_test, bb_gt):
    """
    Calcula IoU entre duas caixas no formato [x1, y1, x2, y2].
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_test + area_gt - wh
    return wh / (union + 1e-8)


def convert_bbox_to_z(bbox):
    """
    Converte [x1, y1, x2, y2] para espaço de estado do Kalman [x, y, s, r]
    x, y: centro da caixa
    s: área (scale)
    r: razão de aspecto (largura / altura)
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Converte o estado do Kalman [x, y, s, r, dx, dy, ds] de volta para [x1, y1, x2, y2].
    """
    w = np.sqrt(np.maximum(0.0, x[2] * x[3]))
    h = x[2] / (w + 1e-6)
    x1 = x[0] - w / 2.0
    y1 = x[1] - h / 2.0
    x2 = x[0] + w / 2.0
    y2 = x[1] + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2]).flatten()
    return np.array([x1, y1, x2, y2, score]).flatten()


class KalmanBoxTracker:
    """
    Rastreia uma única célula convectiva usando Filtro de Kalman.
    Vetor de estado: [x, y, s, r, dx, dy, ds]^T
    """
    count = 0

    def __init__(self, bbox):
        # 7 estados, 4 medições observáveis (x, y, s, r)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # Matriz de Transição de Estado (F) com modelo de velocidade constante
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        # Matriz de Medição (H)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # Alta incerteza inicial de velocidade
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        # Evita expansão irreal da área da célula
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

    def get_velocity(self):
        """Retorna [vx, vy] em pixels por scan."""
        return float(self.kf.x[4]), float(self.kf.x[5])


class SortTracker:
    """
    Rastreador SORT adaptado para rastreamento de convecção/radar.
    """
    def __init__(self, max_age=3, min_hits=2, iou_threshold=0.15):
        """
        max_age: número de frames que o rastro é mantido sem detecção.
        min_hits: detecções mínimas para confirmar a existência do rastro.
        iou_threshold: limiar mínimo de IoU para associar (mais flexível para nuvens fluidas).
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        dets: array numpy de detecções no formato [[x1, y1, x2, y2, score], ...]
        Retorna: array com [[x1, y1, x2, y2, track_id, vx, vy], ...]
        """
        self.frame_count += 1

        # 1. Faz predição da posição atual para todos os rastreadores ativos
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 2. Associa detecções aos rastros existentes usando Algoritmo Húngaro
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks)

        # Atualiza rastreadores associados
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :4])

        # Cria novos rastreadores para detecções não associadas
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :4])
            self.trackers.append(trk)

        # 3. Prepara a saída com os rastros confirmados
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            vx, vy = trk.get_velocity()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # [x1, y1, x2, y2, track_id, vx, vy]
                ret.append(np.concatenate((d, [trk.id + 1, vx, vy])).reshape(1, -1))
            i -= 1
            # Remove rastros muito antigos sem atualização
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 7))

    def _associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

        # Monta matriz de custo IoU
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = compute_iou(det[:4], trk[:4])

        # Algoritmo Húngaro (maximizar IoU = minimizar -IoU)
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.asarray(matched_indices).T

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # Filtra associações com IoU abaixo do limiar
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)