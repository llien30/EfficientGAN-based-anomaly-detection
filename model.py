import torch
import torch.nn as nn

class Encoder(nn.Module):
    '''
    BiGAN Encoder network

    input_size : the image size of the data
    z_dim : the dimention of the latent space
    channel : the number of channels of the data
    nef : the number of filters in encoder
    '''
    def __init__(self, input_size, z_dim, channel, nef, n_extra_layer=0):
        super(Encoder, self).__init__()

        assert input_size%7==0, 'input size has to be a multiple of 7'

        main = nn.Sequential()

        cnef, tisize = nef, 7
        while tisize != input_size:
            cnef = cnef//2
            tisize = tisize*2
        
        main.add_module('initial_Conv-{}-{}'.format(channel, cnef),
                        nn.Conv2d(channel, cnef, kernel_size=3, stride=1, padding=1, bias=False))
                        # the number of stride is the default setting of tf
                        # output size is the same as input_size
        main.add_module('initial_LeakyRU-{}'.format(cnef),
                        nn.LeakyReLU(0.1, inplace=True))
        csize = input_size

        while csize > 7:
            #official kernel_size is 3 but changed to 4
            main.add_module('pyramid_Conv-{}-{}'.format(cnef, cnef*2),
                            nn.Conv2d(cnef, cnef*2, kernel_size=4, stride=2, padding=1, bias=False))
            main.add_module('pyramid_BatchNorm-{}'.format(cnef*2),
                            nn.BatchNorm2d(cnef*2))
            main.add_module('pyramid_LeakyReLU-{}'.format(cnef*2),
                            nn.LeakyReLU(0.1, inplace=True))
            csize = csize//2
            cnef = cnef*2

        for l in range(n_extra_layer):
            main.add_module('extra_Conv-{}-{}'.format(cnef, cnef),
                            nn.Conv2d(cnef, cnef, kernel_size=3, stride=1, padding=1, bias=False))
            main.add_module('extra_BatchNorm-{}'.format(cnef),
                            nn.BatchNorm2d(cnef))
            main.add_module('extra_LeakyReLU-{}'.format(cnef),
                            nn.LeakyReLU(0.1, inplace=True))

        main.add_module('last_linear-{}-{}'.format(cnef*7*7, z_dim),
                        cnef*7*7, z_dim, bias=False)

        self.main = main
    
    def forward(self, input):
        output = self.main(input)

        return output

class NetE(nn.Module):
    '''
    the network of Encoder
    '''
    def __init__(self, CONFIG):
        super(NetE, self).__init__()

        model = Encoder(CONFIG.input_size ,CONFIG.z_dim, CONFIG.channel, CONFIG.nef)
        layers = list(model.main.children())

        self.pyramid = nn.Sequential(*layers[:-1])
        self.linear = nn.Sequential(layers[-1])

    def forward(self, x, CONFIG):
        out = self.pyramid(x)
        out = out.view(-1, CONFIG.nef*7*7) #change to the one dimentional vector
        out = self.linear(out)

        return out
    
class Generator(nn.Module):
    '''
    BiGAN Generator network

    input_size : the image size of the data
    z_dim : the dimention of the latent space
    channel : the number of channels of the image
    ngf : the number of Generator's filter
    '''
    def __init__(self, input_size, z_dim, channel, ngf, n_extra_layer=0):
        super(Generator, self).__init__()

        assert input_size%7==0, 'input size has to be a multiple of 7'

        main = nn.Sequential()

        main.add_module('initial_Linear-{}-{}'.format(z_dim, 1024),
                        nn.Linear(z_dim, 1024, bias=False))
        main.add_module('initial_BatchNorm-{}'.format(1024),
                        nn.BatchNorm1d(1024))
        main.add_module('initial_ReLU-{}'.format(1024),
                        nn.ReLU(inplace=True))

        main.add_module('second_Linear-{}-{}'.format(1024, cngf*7*7),
                        nn.Linear(1024, cngf*7*7, bias=False))
        main.add_module('second_BatchNorm-{}'.format(ngf*7*7),
                        nn.BatchNorm1d(ngf*7*7))
        main.add_module('second_ReLU-{}'.format(ngf*7*7),
                        nn.ReLU(inplace=True))
        csize = 7
        cngf = ngf

        while csize < input_size//2:
            main.add_module('pyramid_Convt-{}-{}'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, kernel_size=4, stride=2, padding=1, bias=False))
            main.add_module('pyramid_BatchNorm-{}'.fomrat(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid_ReLU-{}'.format(cngf//2),
                            nn.ReLU(inplace=True))
            cngf = cngf//2
            csize = csize*2

        for l in range(n_extra_layer):
            main.add_module('extra_Convt-{}-{}'.format(cngf, cngf),
                            nn.ConvTranspose2d(cngf, cngf, kernel_size=3, stride=1, padding=1, bias=False))
            main.add_module('extra_BatchNorm-{}'.format(cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra_ReLU-{}'.format(cngf),
                            nn.ReLU(inplace=True))

        main.add_module('last_Convt-{}-{}'.format(cngf, channel),
                        nn.ConvTranspose2d(cngf, channel, kernel_size=4, stride=2, padding=1, bias=False))
        main.add_module('last_Tanh-{}'.format(channel),
                        nn.Tanh())
        
        self.main = main

    def forward(self, input):
        output = self.main(input)

        return output

class NetG(nn.module):
    '''
    the network of Generator
    '''
    def __init__(self, CONFIG):
        super(NetG, self).__init__()

        model = Generator(CONFIG.input_size, CONFIG.z_dim, CONFIG.channel, CONFIG.ngf)
        layers = list(model.main.children())

        self.linear = nn.Sequential(*layers[:2])
        self.pyramid = nn.Sequential(*layers[2:])

    def forward(self, z, CONFIG):
        out = self.linear(z)
        out = out.view(z.shape[0], ngf, 7, 7) #(batch size, channel, height, width)
        out = self.pyramid(out)

        return out

class Discriminator(nn.Module):
    '''
    BiGAN Discriminator network

    input_size : the image size of the data
    z_dim : the dimention of the latent space
    channel : the number of channels of the image
    ndf : the number of Generator's filter
    '''
    def __init__(self, input_size, z_dim, channel, ndf, n_extra_layer=0):
        super(Discriminator, self).__init__()

        assert input_size%7==0, 'input_size has to be a multiple of 7'

        cndf, tisize = ndf, 7
        while tisize != input_size//2:
            cndf = cndf//2
            tisize = tisize*2
        
        #D(x)
        D_x = nn.Sequential()
        
        D_x.add_module('initial_Conv-{}-{}'.format(channel, cndf),
                        nn.Conv2d(channel, cndf, kernel_size=4, stride=2, padding=1, bias=False))
        D_x.add_module('initial_LeakyReLU-{}'.format(cndf),
                        nn.LeakyReLU(0.1, inplace=True))
        D_x.add_module('initial_Dropout-{}'.format(cndf),
                        nn.Dropout(inplace=True))
        csize = input_size // 2

        while csize > 14:
            D_x.add_module('pyramid_Conv-{}-{}'.format(cndf, cndf*2),
                            nn.Conv2d(cndf, cndf*2, kernel_size=4, stride=2, padding=1, bias=False))
            D_x.add_module('pyramid_LeakyReLU-{}'.format(cndf*2),
                            nn.LeakyaReLU(0.1, inplace=True))
            D_x.add_module('pyramid_Dropout-{}'.format(cndf*2),
                            nn.Dropout(inplace=True))
            csize = csize//2
            cndf = cndf*2

        for l in range(n_extra_layer):
            D_x.add_module('extra_Conv-{}-{}'.format(cndf, cndf),
                            nn.Conv2d(cndf, cndf, kernel_size=3, stride=1, padding=1, bias=False))
            D_x.add_module('extra_LeakyReLU-{}'.format(cndf),
                            nn.LeakyReLU(0.1, inplace=True))
            D_x.add_module('extra_Dropout-{}'.format(cndf),
                            nn.Dropout(inplace=True))

        D_x.add_module('last_Conv-{}-{}'.format(cndf, cndf),
                        nn.Conv2d(cndf, cndf, kernel_size=4, stride=2, padding=1, bias=False))
        D_x.add_module('last_LeakyReLU-{}'.format(cndf),
                        nn.LeakyReLU(0.1, inplace=True))
        D_x.add_module('pyramid_Dropout-{}'.format(cndf),
                            nn.Dropout(inplace=True))

        #D(z)
        D_z = nn.Sequential()

        D_z.add_module('z_Linear',
                        nn.Linear(z_dim, 512))
        D_z.add_module('z_LeakyReLU',
                        nn.LeakyReLU(0.1, inplace=True))
        D_z.add_module('z_Dropout',
                        nn.Dropout(inplace=True))

        #D(x,z)
        D_xz = nn.Sequential()
        D_xz.add_module('concat_Linear-{}-{}'.format(512+cndf*7*7, 1024),
                        nn.Linear(512+cndf*7*7, 1024))
        D_xz.add_module('concat_LeakyReLU-{}'.format(1024),
                        nn.LeakyReLU(0.1, inplace=True))
        D_xz.add_module('concat_Dropout-{}'.format(1024),
                        nn.Dropout(inplace=True))

        D_xz.add_module('last_Linear-{}-{}'.format(1024, 1),
                        nn.Linear(1024, 1))
        

        self.D_x = D_x
        self.D_z = D_z
        self.D_xz = D_xz

    def forward(self, x, z, y):
        x_out = D_x(x)
        z_out = D_z(z)
        output = D_xz(y)
        return output

class NetD(nn.Module):
    '''
    the network of Discriminator
    '''
    def __init__(self, CONFIG):
        super(NetD, self).__init__()
        model = Discriminator(CONFIG.input_size, CONFIG.z_dim, CONFIG.channel, CONFIG.ndf)
        #D(x)
        layer_x = list(model.D_x.children())
        self.layer_x = nn.Sequential(layer_x)
        #D(z)
        layer_z = list(model.D_z.children())
        self.layer_z = nn.Sequenrial(layer_z)
        #D(x,z)
        layer = list(model.D_xz.children())
        self.feature = nn.Sequential(*layer[:-1])
        self.classifier = nn.Sequential(layer[-1])

    def forward(self, x, z, CONFIG):
        x_out = self.layer_x(x)
        x_out = out.view(-1, CONFIG.ndf*7*7)

        z_out = self.layer_z(z)

        y = torch.cat([x_out, z_out], dim=1)
        out = self.feature(y)

        feature = out
        feature = feature.view(feature.size()[0], -1)

        our = self.classifier(out)

        return out, feature