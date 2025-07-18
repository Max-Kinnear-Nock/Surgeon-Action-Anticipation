import torch
from torch.autograd import Variable

from Utils.frameUtils import extract_frames

class Trainer:
    def __init__(self, model, dataset, m1, m2, recognition_length, anticipation_length, use_gpu=True):
        self.model = model
        self.dataset = dataset
        self.m1 = m1
        self.m2 = m2
        self.recognition_length = recognition_length
        self.anticipation_length = anticipation_length
        self.use_gpu = use_gpu
        self.sig_f = torch.nn.Sigmoid()

    def run_step(self, data, optimizer=None, train=True):
        if train:
            self.model.train()
            optimizer.zero_grad()
        else:
            self.model.eval()

        # Move data to GPU
        data = [Variable(d.cuda()) if self.use_gpu else Variable(d) for d in data]
        frames, labels_1, labels_2, labels_3, labels_4 = data

        labels_recog, labels_anticp = extract_frames(
            labels=[labels_1, labels_2, labels_3, labels_4],
            recognition_length=self.recognition_length,
            anticipation_length=self.anticipation_length
        )

        anticp_output, recog_output, loss_anticp, loss_recog, MC_loss = self.model(
            frames, labels_recog, labels_anticp, self.dataset.pos_weights
        )

        total_mc_loss_0 = sum(torch.mean(MC_loss[i][0]) for i in range(4))
        total_mc_loss_1 = sum(torch.mean(MC_loss[i][1]) for i in range(4))

        loss = sum(loss_recog) + sum(loss_anticp) + self.m1 * total_mc_loss_0 + self.m2 * total_mc_loss_1

        if train:
            loss.backward()
            optimizer.step()

        return loss.item(), sum(loss_recog).item(), sum(loss_anticp).item(), recog_output, anticp_output, labels_recog, labels_anticp

    def train_model(self, loader, optimizer, recognize, anticipate):
        total, recog, anticp = 0.0, 0.0, 0.0
        num_batches_train = len(loader)
        batch_progress = 0

        for data in loader:
            loss_total, loss_recog, loss_anticp, recog_out, anticp_out, lab_recog, lab_anticp = \
                self.run_step(data, optimizer, train=True)

            total += loss_total
            recog += loss_recog
            anticp += loss_anticp

            recognize.update(lab_recog[3].detach().cpu().float(), self.sig_f(recog_out[3].detach()).cpu().float())
            anticipate.update(lab_anticp[3].detach().cpu().float(), self.sig_f(anticp_out[3].detach()).cpu().float())

            batch_progress += 1
            if batch_progress >= num_batches_train:
                percent = 100.0
                print(f'Batch progress: {percent}% [{num_batches_train}/{num_batches_train}]', end='\n')
            else:
                percent = round(batch_progress / num_batches_train * 100, 2)
                print(f'Batch progress: {percent}% [{batch_progress}/{num_batches_train}]', end='\r')

        return total, recog, anticp

    def validate_model(self, loader, recognize, anticipate):
        return self._eval(loader, recognize, anticipate)

    def test_model(self, loader, recognize, anticipate):
        return self._eval(loader, recognize, anticipate)

    def _eval(self, loader, recognize, anticipate):
        total, recog, anticp = 0.0, 0.0, 0.0
        num_batches_validation = len(loader)
        batch_progress = 0 

        with torch.no_grad():
            for data in loader:
                loss_total, loss_recog, loss_anticp, recog_out, anticp_out, lab_recog, lab_anticp = \
                    self.run_step(data, train=False)

                total += loss_total
                recog += loss_recog
                anticp += loss_anticp

                recognize.update(lab_recog[3].detach().cpu().float(), self.sig_f(recog_out[3].detach()).cpu().float())
                anticipate.update(lab_anticp[3].detach().cpu().float(), self.sig_f(anticp_out[3].detach()).cpu().float())

                batch_progress += 1
                if batch_progress >= num_batches_validation:
                    percent = 100.0
                    print(f'Batch progress: {percent}% [{num_batches_validation}/{num_batches_validation}]', end='\n')
                else:
                    percent = round(batch_progress / num_batches_validation * 100, 2)
                    print(f'Batch progress: {percent}% [{batch_progress}/{num_batches_validation}]', end='\r')
                    
        return total, recog, anticp