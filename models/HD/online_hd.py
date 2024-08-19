class HD_model():
    def __init__(self, classes = 20, d = 2500, num_features=(409, 204, 153), lr = 0.01, **kwargs):
        self.d = d
        self.div = kwargs['div']
        self.device = kwargs['device']
        self.classes_hv = torch.zeros((classes, self.d))
        self.flatten = nn.Flatten(0,1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.softmax_2 = torch.nn.Softmax(dim=2)
        self.random_projection_0 = torchhd.embeddings.Projection(num_features[0], self.d, device=kwargs['device'])
        self.random_projection_1 = torchhd.embeddings.Projection(num_features[1], self.d, device=kwargs['device'])
        self.random_projection_2 = torchhd.embeddings.Projection(num_features[2], self.d, device=kwargs['device'])
        self.stages = torchhd.random(3, d, device=kwargs['device'])
        #self.random_projection = {0:self.random_projection_0, 1:self.random_projection_1, 2:self.random_projection_2,}
        #self.random_projection = (self.random_projection_0, self.random_projection_1, self.random_projection_2)
        #self.random_projection = BatchProjection(3, num_features[0], self.d, device=kwargs['device'])
        #self.random_projection_global = torchhd.embeddings.Projection(num_features, self.d)
        self.lr = lr
        self.num_features = num_features

    def to(self, *args):
        self.classes_hv = self.classes_hv.to(*args)
        self.random_projection_0 = self.random_projection_0.to(*args)
        self.random_projection_1 = self.random_projection_1.to(*args)
        self.random_projection_2 = self.random_projection_2.to(*args)
        self.random_projection = {0:self.random_projection_0, 1:self.random_projection_1, 2:self.random_projection_2}
        #self.random_projection = self.random_projection.to(*args)

    def encode(self, input_x, infer=False):
        #print(input_x.get_device())
        #hv_0 = self.random_projection(input_x) # <-- BATCH
        #print(hv_0.shape) # (3,#,d)
        hv_0 = torchhd.bind(self.random_projection[0](input_x[0]).sign(), self.stages[0])
        hv_1 = torchhd.bind(self.random_projection[1](input_x[1]).sign(), self.stages[1])
        hv_2 = torchhd.bind(self.random_projection[2](input_x[2]).sign(), self.stages[2])
        hv_all = torch.stack((hv_0, hv_1, hv_2))
        if infer:
            hv_all = torch.sum(hv_all, dim=0).sign()

        #x = input("Enter")

        return hv_all
    
    def forward(self, input_h):
        #print(input_h.shape)
        #print(input_h[1,:,:self.num_features[1]].shape)
        input_h = (input_h[0], input_h[1,:,:self.num_features[1]], input_h[2,:,:self.num_features[2]])
        hv = self.encode(input_h)
        #hv = torch.sum(hv, dim=0).sign()
        sim = self.similarity(hv, True)
        #print("sim: ", sim.shape)
        best_ind = torch.argmax(sim, dim=2)
        #print(best_ind.shape)
        best_sim = torch.max(sim, dim=2).values
        #print(best_sim.shape)
        best_sim_2 = torch.argmax(best_sim, dim=0)
        #print(torch.bincount(best_sim_2))
        #print(best_sim_2.shape)
        pred_label = best_ind[best_sim_2, torch.arange(best_ind.shape[1])]
        #print("ALL ", pred_label)
        hv = torch.sum(hv, dim=0).sign() # hv = hv[best_sim_2, torch.arange(pred_label.shape[0])] <- Best just gets 0
        #print("hv", hv.shape)
        #print("sim: ", sim[best_sim_2, torch.arange(sim.shape[1])].shape)
        return hv, sim[best_sim_2, torch.arange(sim.shape[1])], pred_label
        
    def similarity(self, point, group=False):
        sim = torchhd.cosine_similarity(point, self.classes_hv)
        if group:
            sim = self.softmax_2(sim)
        else:
            sim = self.softmax(sim)

        return sim
    
    def train(self, input_points, classification, **kwargs):
        #print(input_points.shape)
        for idx in torch.arange(input_points.shape[1]).chunk(self.div):
            hv_all, sim_all, pred_labels = self.forward(input_points[:, idx, :])
            idx = idx.to(self.device)
            class_batch = classification[idx].type(torch.LongTensor).to(self.device)
            novelty = 1 - sim_all[torch.arange(idx.shape[0]), class_batch]
            updates = hv_all.transpose(0,1)*torch.mul(novelty, self.lr) # Normal HD with novelty
            updates = updates.transpose(0,1)
            
            # Update all of the classes with the actual label
            self.classes_hv.index_add_(0, class_batch, updates)
            
            #Substract when class is different then actual
            
            #if (pred_labels != c):
            #    self.classes_hv[pred_labels] += -1*hv_all*self.lr*(1-sim_all[pred_labels])
            
            
            # ONLINEHD
            mask_dif = class_batch != pred_labels
            novelty = 1 - sim_all[mask_dif, pred_labels[mask_dif]] # only the ones updated
            updates = hv_all[mask_dif].transpose(0,1)*torch.mul(novelty, self.lr)
            updates = torch.mul(updates, -1)
            updates = updates.transpose(0,1)
            updates_2 = torch.zeros((idx.shape[0], self.d), device=self.device) # all zeros original
            updates_2[mask_dif] = updates # update vectors for the ones that changed

            self.classes_hv.index_add_(0, pred_labels, updates_2)