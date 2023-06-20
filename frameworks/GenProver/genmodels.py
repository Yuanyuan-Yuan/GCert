try:
    from . import components as n
    from . import ai
    from . import scheduling as S
    from . import helpers as h
except:
    import components as n
    import scheduling as S
    import ai
    import helpers as h

def ConvTinyInv(c):
    w = int(c[-1] / 2)
    return n.Seq(n.InvLeNet([200, w * w * 8], w, [ (8,3,2,1,1) , (h.product(c[:-2]),3,1,1,0) ], ibp_init=True), n.View(c))


def ConvSmallInv(c):
    w = int(c[-1] / 2)
    return n.Seq(n.InvLeNet([400, w * w * 8], w, [ (16,3,2,1,1) , (h.product(c[:-2]),3,1,1,0) ], ibp_init=True), n.View(c))

def ConvMedInv(c):
    w = int(c[-1] / 4)
    return n.Seq(n.InvLeNet([1000,2000, w * w * 16], w, [ (64,3,2,1,1) , (32,3,1,1,0), (32,3,2,1,1) , (h.product(c[:-2]),3,1,1,0) ], ibp_init = True), n.View(c))


def ConvLargeInv(c):
    w = int(c[-1] / 4)
    return n.Seq(n.InvLeNet([500, 1000,2000, w * w * 16], w, [ (128,3,2,1,1) , (64,3,1,1,0), (32,3,2,1,1) , (h.product(c[:-2]),3,1,1,0) ], ibp_init = True), n.View(c))


def ConvDCInv(c):
    w = int(c[-1] / 4)
    return n.Seq(n.InvLeNet([w * w * 16], w, [ (128,3,2,1,1) , (64,3,1,1,0), (32,3,2,1,1) , (h.product(c[:-2]),3,1,1,0) ], ibp_init = True), n.View(c))

def ConvDeepCInv(c):
    w = int(c[-1] / 8)
    return n.Seq(n.InvLeNet([w * w * 16], w, [ (256,3,2,1,1) , (128,3,1,1,0), (64,3,2,1,1) , (32,3,1,1,0), (16,3,2,1,1) , (h.product(c[:-2]),3,1,1,0) ], ibp_init = True), n.View(c))


def ConvInvTest(c=[1, 32, 32]):
    w = int(c[-1] / 4)
    return n.Seq(
        n.InvLeNet(
            [1000, 2000, w * w * 16],
            w,
            [
                (64,3,2,1,1), (32,3,1,1,0), (32,3,2,1,1),
                (h.product(c[:-2]),3,1,1,0)
            ],
            ibp_init=True
        ),
        n.View(c)
        )

def FFNN(layers, last_lin = False, last_zono = False, **kargs):
    starts = layers
    ends = []
    if last_lin:
        ends = (
            [CorrelateAll(only_train=False)] if last_zono else []
        ) + [
                PrintActivation(activation = "Affine"),
                Linear(layers[-1], **kargs)
        ]
        
        starts = layers[:-1]
    
    return Seq(
        *(
            [ 
                Seq(
                    PrintActivation(**kargs),
                    Linear(s, **kargs),
                    activation(**kargs)
                ) for s in starts
            ] + ends
        )
    )

def InvLeNet(ly, w, conv_layers, bias=True, normal=False, **kargs):
    def transfer(tp, lin = False):
        return (ConvTranspose2D if lin else ConvTranspose)(
                                        out_channels=tp[0],
                                        kernel_size=tp[1],
                                        stride=tp[2],
                                        padding=tp[3],
                                        out_padding=tp[4],
                                        bias=False,
                                        normal=normal,
                                        **kargs
                                    )
                      
    return Seq(
        FFNN(ly, bias=bias, **kargs),
        Unflatten2d(w), 
        *[
            transfer(s) for s in conv_layers[:-1]
        ],
        transfer(conv_layers[-1], lin=True)
    )

def dcgan_upconv(nin, nout, **kargs):
    return n.Seq(
            n.ConvTranspose2D(nin, nout, 4, 2, 1),
            n.BatchNorm(nout),
            n.ReLU(),
        )


def ConvGenerator(**kargs):
    nf = 4
    dim = 50
    nc = 1
    return n.Seq(
        n.View([dim, 1, 1]),
        # n.ConvTranspose(out_channels=nf * 4, kernel_size=4, stride=1, padding=0, activation='ReLU', batch_norm=True, **kargs),
        n.Seq(
            n.ConvTranspose2D(out_channels=nf * 4, kernel_size=4, stride=1, padding=0, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        # n.ConvTranspose(out_channels=nf * 2, kernel_size=4, stride=2, padding=1, activation='ReLU', batch_norm=True, **kargs),
        n.Seq(
            n.ConvTranspose2D(out_channels=nf * 2, kernel_size=4, stride=2, padding=1, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        # n.ConvTranspose(out_channels=nf, kernel_size=4, stride=2, padding=1, activation='ReLU', batch_norm=True, **kargs),
        n.Seq(    
            n.ConvTranspose2D(out_channels=nf, kernel_size=4, stride=2, padding=1, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.Seq(
            n.ConvTranspose2D(out_channels=nc, kernel_size=4, stride=2, padding=1, **kargs),
            n.Activation(activation='ReLU', **kargs),
            n.Negate(**kargs),
            n.AddOne(**kargs),
            n.Activation(activation='ReLU', **kargs)
        )
    )

def ConvGenerator32(**kargs):
    nf = 4
    dim = 50
    nc = 1
    return n.Seq(
        n.View([dim, 1, 1]),
        # n.ConvTranspose(out_channels=nf * 4, kernel_size=4, stride=1, padding=0, activation='ReLU', batch_norm=True, **kargs),
        n.Seq(
            n.ConvTranspose2D(out_channels=nf * 4, kernel_size=4, stride=1, padding=0, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        # n.ConvTranspose(out_channels=nf * 2, kernel_size=4, stride=2, padding=1, activation='ReLU', batch_norm=True, **kargs),
        n.Seq(
            n.ConvTranspose2D(out_channels=nf * 2, kernel_size=4, stride=2, padding=1, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        # n.ConvTranspose(out_channels=nf, kernel_size=4, stride=2, padding=1, activation='ReLU', batch_norm=True, **kargs),
        n.Seq(    
            n.ConvTranspose2D(out_channels=nf, kernel_size=4, stride=2, padding=1, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.Seq(
            n.ConvTranspose2D(out_channels=nc, kernel_size=4, stride=2, padding=1, **kargs),
            n.Activation(activation='ReLU', **kargs),
            n.Negate(**kargs),
            n.AddOne(**kargs),
            n.Activation(activation='ReLU', **kargs)
        )
    )

def Recog32(**kargs):
    nf = 16
    dim = 100
    return n.Seq(
        n.CatTwo(),
        n.Seq(
            n.Conv2D(out_channels=nf, kernel_size=4, stride=2, padding=1, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.Seq(
            n.Conv2D(out_channels=nf * 2, kernel_size=4, stride=2, padding=1, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.Seq(
            n.Conv2D(out_channels=nf * 4, kernel_size=4, stride=2, padding=1, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.Seq(
            n.Conv2D(out_channels=dim, kernel_size=4, stride=1, padding=0, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.View([dim]),
        n.Seq(
            n.Linear(1, **kargs),
            n.Activation(activation='Sigmoid', **kargs),
        ),
    )

def F1(**kargs):
    n_class = 1
    dim = 10
    return n.Seq(
        n.View([32 * 32]),
        n.Seq(
            n.Linear(dim, **kargs),
            # n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.Seq(
            n.Linear(dim, **kargs),
            # n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.Seq(
            n.Linear(n_class, **kargs),
        ),
    )

def F2(**kargs):
    n_class = 1
    dim = 10
    nf = 16
    return n.Seq(
        n.Seq(
            n.Conv2D(out_channels=nf, kernel_size=4, stride=2, padding=1, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.Seq(
            n.Conv2D(out_channels=nf * 2, kernel_size=4, stride=4, padding=0, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.View([nf * 2 * 4 * 4]),
        n.Seq(
            n.Linear(n_class, **kargs),
        ),
    )

def F3(**kargs):
    n_class = 1
    dim = 10
    nf = 16
    return n.Seq(
        n.Seq(
            n.Conv2D(out_channels=nf, kernel_size=4, stride=2, padding=1, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.Seq(
            n.Conv2D(out_channels=nf * 2, kernel_size=4, stride=2, padding=1, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.Seq(
            n.Conv2D(out_channels=nf * 4, kernel_size=4, stride=2, padding=1, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.Seq(
            n.Conv2D(out_channels=dim, kernel_size=4, stride=1, padding=0, **kargs),
            n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.View([dim]),
        n.Seq(
            n.Linear(dim, **kargs),
            # n.BatchNorm(training=False, **kargs),
            n.Activation(activation='ReLU', **kargs),
        ),
        n.Seq(
            n.Linear(n_class, **kargs),
        ),
    )
