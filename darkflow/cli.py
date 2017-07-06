import tensorflow as tf

from .defaults import argHandler #Import the default arguments
import os
from .net.build import TFNet

def cliHandler(args):
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    # make sure all necessary dirs exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)
    _get_dir([FLAGS.imgdir, FLAGS.binary, FLAGS.backup, 
             os.path.join(FLAGS.imgdir,'out'), FLAGS.summary])

    # fix FLAGS.load to appropriate type
    try: FLAGS.load = int(FLAGS.load)
    except: pass

    if FLAGS.jobType == 'ps' or FLAGS.jobType == 'worker' :
        print('Distributed Training')
        ps_hosts = FLAGS.psHosts.split(',')
        print('PS hosts are: %s' % ps_hosts)
        worker_hosts = FLAGS.workerHosts.split(',')
        print('Worker hosts are: %s' % worker_hosts)
        print('taskId is : %d' % FLAGS.taskId)
        cluster = tf.train.ClusterSpec({'ps': ps_hosts,
                                        'worker': worker_hosts})
        num_of_workers = len(cluster.as_dict()['worker'])
        server = tf.train.Server(
            cluster,
            job_name=FLAGS.jobType,
            task_index=FLAGS.taskId)
        
        if FLAGS.jobType == 'ps' :
            print('Running as Parameter Server');
            server.join()
            return
        
        tfnet = TFNet(FLAGS, cluster = cluster, replicas_to_aggregate = num_of_workers, num_of_workers = num_of_workers, server = server)
    else:
        tfnet = TFNet(FLAGS)
    
    if FLAGS.demo:
        tfnet.camera()
        exit('Demo stopped, exit.')

    if FLAGS.train:
        print('Enter training ...'); tfnet.train()
        if not FLAGS.savepb: 
            exit('Training finished, exit.')

    if FLAGS.savepb:
        print('Rebuild a constant version ...')
        tfnet.savepb(); exit('Done')

    tfnet.predict()
