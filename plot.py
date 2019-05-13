import pickle
K = list(range(50,1851,50))
EVAL = []
for k in K:
	f = open('result_v2_'+str(k), 'r')
	predictions = [i.split() for i in f.readlines()]
	f.close()
	n = len(predictions)

	TP = len([i for i in range(n) if predictions[i][1] == '+1' and predictions[i][0] == '+1'])
	FN = len([i for i in range(n) if predictions[i][1] == '-1' and predictions[i][0] == '+1'])
	FP = len([i for i in range(n) if predictions[i][1] == '+1' and predictions[i][0] == '-1'])
	TN = len([i for i in range(n) if predictions[i][1] == '-1' and predictions[i][0] == '-1'])
	print(TP,FN,FP,TN)
	recall = TP/(TP+FN)
	precision = TP/(TP+FP)
	accuracy = (TP+TN)/n
	error_rate = 1-accuracy
	F1 = 2 * precision * recall/(precision+recall)
	EVAL.append([recall,precision,accuracy,error_rate,F1])

pickle.dump({'EVAL':EVAL,'K':K}, open("result_v2.pkl", "wb"), True)
print('all done')