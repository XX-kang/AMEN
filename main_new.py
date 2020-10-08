import train
import cam_save
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
devices_ids = 0

def adjust_alpha(alpha, model_arch):
    print('1')
    train.main(data_path='data/BreakHis/200X/', data_name='BK_200X_img', channels=3,
               classes=8, run_id_old='BK_200X_img_1000', data_name_old='BK_200X_img',
               model_arch= model_arch)
    cam_save.main(run_id='BK_200X_img_1001', classes=8, split='train',alpha = alpha)
    cam_save.main(run_id='BK_200X_img_1001', classes=8, split='test',alpha = alpha)
    print('2')
    train.main(data_path='outputs/BK_200X_img_1001/result1/', data_name='BK_200X_result1_roi',
               channels=3, classes=8, run_id_old='BK_200X_img_1001', data_name_old='BK_200X_img'
               , model_arch= model_arch)
    cam_save.main(run_id='BK_200X_result1_roi_1002', classes=8, split='train',alpha = alpha)
    cam_save.main(run_id='BK_200X_result1_roi_1002', classes=8, split='test',alpha = alpha)
    print('3')
    train.main(data_path='outputs/BK_200X_result1_roi_1002/result1/',
               data_name='BK_200X_result1_part',
               channels=3, classes=8, run_id_old='BK_200X_result1_roi_1002',
               data_name_old='BK_200X_result1_roi', model_arch= model_arch)
    cam_save.main(run_id='BK_200X_result1_part_1003', classes=8, split='train', alpha=alpha)
    cam_save.main(run_id='BK_200X_result1_part_1003', classes=8, split='test', alpha=alpha)
    print('4')
    train.main(data_path='outputs/BK_200X_result1_part_1003/result1/',
               data_name='BK_200X_result1_fusion',
               channels=3, classes=8, run_id_old='BK_200X_result1_roi_1003',
               data_name_old='BK_200X_result1_part', model_arch= model_arch)

if __name__ == "__main__":
    import shutil

    #0.01, 0.005, 0.0001, 0.0005,0.00001, 0.00005, 0.000001
    alpha = [ 0.01, 0.005, 0.0001, 0.0005 ]
    #model_arch = ['googlenet','densenet121','resnet50','resnet101']
    model_arch = ['vgg16_cam','vgg19_cam']
    for i in alpha:
        for m in model_arch:
            import csv
            csvFile = open("record.csv", "a")  # 创建csv文件
            writer1 = csv.writer(csvFile)  # 创建写的对象
            writer1.writerow([i, '200X'])
            csvFile.close()
            if os.path.exists('runs'):
                shutil.move('runs', os.path.join('vision','200X',m, 'runs', str(i)))
            os.makedirs('runs')
            if os.path.exists('pkls'):
                shutil.move('pkls', os.path.join('vision','200X',m,'pkls', str(i)))
            os.makedirs('pkls')
            if os.path.exists('outputs'):
                shutil.move('outputs', os.path.join('vision','200X',m, 'outputs', str(i)))
            os.makedirs('outputs')
            adjust_alpha(i,m)
    #cam_save.main(run_id='BK_200X_img_1001', classes=8, split='train',alpha = 0.0001)