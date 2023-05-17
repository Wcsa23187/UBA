import numpy as np
import torch


def evaluate(model, top_k, train_dict, vt_dict, valid_dict, item_num, batch_size, flag):
    recommends = []
    for i in range(len(top_k)):
        recommends.append([])
    # print("recommender %s" %(recommends))
    with torch.no_grad():
        pred_list_all = []
        # print("vt_dict.keys() %s" % (vt_dict.keys()))
        for i in vt_dict.keys():  # for each user
            #  print("i %s" % (i))
            # print("vt_dict[i] %s" % (vt_dict[i]))
            if len(vt_dict[i]) != 0:  # if
                prediction_list = []
                # print("item_num %s" % (item_num))
                # print("batch_size %s" % (batch_size))
                for start_idx in range(0, item_num, batch_size):  # batch size is batch item num
                    # print("start_idx %s" % (start_idx))
                    end_idx = min(start_idx + batch_size, item_num)
                    # print("end_idx %s" % (end_idx))
                    user = torch.full((end_idx - start_idx,), i, dtype=torch.int64).cuda()  # batch_size, same user

                    item = torch.arange(start_idx, end_idx, dtype=torch.int64).cuda()

                    prediction = model(user, item)
                    # print("prediction %s" % (prediction))
                    prediction_tmp = prediction.detach().cpu().numpy().tolist()
                    prediction_list.extend(prediction_tmp)
                for j in train_dict[i]:  # mask train
                    prediction_list[j] -= float('inf')
                if flag == 1:  # mask validation
                    if i in valid_dict:
                        for j in valid_dict[i]:
                            prediction_list[j] -= float('inf')
                pred_list_all.append(prediction_list)
        # print("pred_list_all %s" % (pred_list_all))
        predictions = torch.Tensor(pred_list_all).cuda()  # shape: (n_user,n_item)
        # print("predictions %s" % (predictions))
        for idx in range(len(top_k)):
            _, indices = torch.topk(predictions, int(top_k[idx]))
            recommends[idx].extend(indices.tolist())
    # print("recommends %s" % (recommends))
    return recommends


def metrics(model, top_k, train_dict, vt_dict, valid_dict, item_num, batch_size, flag):
    precision, recall, NDCG, MRR = [], [], [], []
    recommends = evaluate(model, top_k, train_dict, vt_dict, valid_dict, item_num, batch_size, flag)
    for idx in range(len(top_k)):
        sumForPrecision, sumForRecall, sumForNDCG, sumForMRR = 0, 0, 0, 0
        k = -1
        for i in vt_dict.keys():  # for each user
            if len(vt_dict[i]) != 0:
                k += 1  # to let we know the user order in topk 10 20 50  just like i
                userhit = 0
                dcg = 0
                idcg = 0
                idcgCount = len(vt_dict[i])  # user 's item in test or vali dict
                ndcg = 0
                mrrFlag = True
                userMRR = 0
                # recommends[idx][k] the user k's item
                for index, thing in enumerate(recommends[idx][k]):
                    # pdb.set_trace()
                    if thing in vt_dict[i]:
                        userhit += 1
                        dcg += 1.0 / (np.log2(index + 2))
                        if mrrFlag:
                            userMRR = 1.0 / (index + 1)
                            mrrFlag = False
                    if idcgCount > 0:
                        idcg += 1.0 / (np.log2(index + 2))
                        idcgCount -= 1
                if (idcg != 0):
                    ndcg += (dcg / idcg)

                sumForPrecision += userhit / len(recommends[idx][k])
                sumForRecall += userhit / len(vt_dict[i])
                sumForNDCG += ndcg
                sumForMRR += userMRR

        precision.append(round(sumForPrecision / len(vt_dict.keys()), 4))
        recall.append(round(sumForRecall / len(vt_dict.keys()), 4))
        NDCG.append(round(sumForNDCG / len(vt_dict.keys()), 4))
        MRR.append(round(sumForMRR / len(vt_dict.keys()), 4))
    # compute for the target user
    precision_user, recall_user, NDCG_user, MRR_user = [], [], [], []

    '''
    user = [646, 2975, 2597, 5678, 5682, 4146, 4918, 1082, 3386, 317, 4679, 1238, 2398, 869, 5862, 4711, 3305, 623, 374,
            123]
    '''
    # user = [5, 103, 153, 228, 286, 317, 362, 374, 513, 587, 1040, 1061, 1176, 1238, 1338, 2340, 2398, 2816, 3305, 3386, 3582, 3628, 3903, 4146, 4330, 4364, 4399, 4572, 4586, 4711, 4926, 5017, 5345, 5618, 5682, 35, 239, 623, 970, 1802, 2191, 2678, 2704, 3734, 4434, 4445, 4679, 4770, 4968, 5862, 4332, 4975, 5696, 1771, 425, 646, 789, 797, 825, 1082, 2315, 2347, 2365, 3987, 4118, 4290, 4667, 4918, 4962, 5678, 123, 1059, 3450, 3013, 208, 1160, 3532, 5854, 4738, 1361, 1462, 1509, 5520, 160, 2101, 468, 2597, 869, 2975]
    # best users 
    # user = [5520, 35, 5678, 160, 1771, 1509, 4738, 825, 317, 2398, 4962, 2365, 1338, 4975, 2704, 970, 5682, 3305, 5618, 4399]
    # sample 50 from 89 suitable users
    # user = [513, 1160, 2315, 4364, 2191, 4118, 3734, 1176, 797, 2975, 2340, 1061, 2597, 425, 4399, 5682, 2101, 1462, 825, 1338, 3386, 4667, 317, 4926, 3903, 5696, 4290, 4679, 587, 3532, 208, 1361, 4434, 468, 1238, 4572, 4445, 5345, 1509, 5862, 103, 4968, 3305, 4586, 4332, 4975, 239, 2678, 3450, 123]
    # user = [2816, 4738, 5, 646, 4364, 2191, 5520, 1040, 3987, 789, 3734, 4118, 5017, 286, 2975, 4770, 1059, 35, 2340, 2597, 2347, 3628, 5678, 2101, 1462, 825, 1082, 4667, 1338, 317, 3903, 3013, 970, 587, 208, 1238, 4445, 5854, 869, 5862, 103, 4968, 3305, 362, 4332, 4975, 623, 5618, 2678, 123]
    # user = [513, 646, 1040, 5017, 2975, 1061, 2597, 2347, 4146, 2101, 1462, 4667, 4926, 3903, 3013, 4679, 208, 4434, 468, 1238, 4572, 4445, 4968, 4330, 4586, 4332, 239, 374, 3450, 3582,5520 ,35 ,5678 ,160  ,1771 ,1509 ,4738 ,825 ,317 , 2398 ,4962, 2365 ,1338, 4975, 2704 ,970 , 5682, 3305 ,5618 ,4399]
    user = [5520 ,5678   ,1771  ,4738  ,317  ,4962 ,1338, 4975 ,970 , 3305  ,5, 646, 1802, 2191, 2704, 3987, 789, 3734, 5017, 797, 286, 160, 4770, 35, 2340, 2597, 425, 2347, 5682, 4918, 1462, 3386, 1082, 2365, 4926, 3903, 587, 3532, 4434, 1238, 4572, 5854, 5345, 1509, 362, 4330, 623, 5618, 2678, 3582]
    for idx in range(len(top_k)):
        print("*****************************")
        sumForPrecision, sumForRecall, sumForNDCG, sumForMRR = 0, 0, 0, 0
        k = -1
        for i in user:  # for each user
            print("###########")
            print('the target user is %s' % i)
            print(vt_dict[i])
            if len(vt_dict[i]) != 0:
                k += 1  # to let we know the user order in topk 10 20 50  just like i
                userhit = 0
                dcg = 0
                idcg = 0
                idcgCount = len(vt_dict[i])  # user 's item in test or vali dict
                ndcg = 0
                mrrFlag = True
                userMRR = 0
                # recommends[idx][k] the user k's item
                for index, thing in enumerate(recommends[idx][k]):
                    # pdb.set_trace()
                    if thing in vt_dict[i]:
                        userhit += 1
                        dcg += 1.0 / (np.log2(index + 2))
                        if mrrFlag:
                            userMRR = 1.0 / (index + 1)
                            mrrFlag = False
                    if idcgCount > 0:
                        idcg += 1.0 / (np.log2(index + 2))
                        idcgCount -= 1
                if (idcg != 0):
                    ndcg += (dcg / idcg)
                print("the recall %s" % (userhit / len(vt_dict[i])))
                print('the ndcg %s' % ndcg)
                sumForPrecision += userhit / len(recommends[idx][k])
                sumForRecall += userhit / len(vt_dict[i])
                sumForNDCG += ndcg
                sumForMRR += userMRR

        precision_user.append(round(sumForPrecision / len(user), 4))
        recall_user.append(round(sumForRecall / len(user), 4))
        NDCG_user.append(round(sumForNDCG / len(user), 4))
        MRR_user.append(round(sumForMRR / len(user), 4))
        print("output the data of the target user")
        print(precision_user)
        print(recall_user)
        print(NDCG_user)
        print(MRR_user)

    return precision, recall, NDCG, MRR


def print_epoch_result(precision, recall, NDCG, MRR):
    print("Precision: {} Recall: {} NDCG: {} MRR: {}".format(
        '-'.join([str(x) for x in precision]),
        '-'.join([str(x) for x in recall]),
        '-'.join([str(x) for x in NDCG]),
        '-'.join([str(x) for x in MRR])))


def print_results(loss, valid_result, test_result):
    """output the evaluation results."""
    if loss is not None:
        print("[Train]: loss: {:.4f}".format(loss))
    if valid_result is not None:
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
            '-'.join([str(x) for x in valid_result[0]]),
            '-'.join([str(x) for x in valid_result[1]]),
            '-'.join([str(x) for x in valid_result[2]]),
            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None:
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
            '-'.join([str(x) for x in test_result[0]]),
            '-'.join([str(x) for x in test_result[1]]),
            '-'.join([str(x) for x in test_result[2]]),
            '-'.join([str(x) for x in test_result[3]])))
