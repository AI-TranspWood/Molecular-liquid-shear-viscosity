import io
import sys

from aiida import orm
from aiida.cmdline.params import arguments, types
import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from . import cmd_root


@cmd_root.command('plot-viscosity')
@arguments.DATUM(
    'data',
    required=True,
    type=types.DataParamType(sub_classes=('aiida.data:core.array',))
)
@click.option(
    '-s', '--show-plot',
    is_flag=True,
    default=False,
    help='Show the plot interactively.'
)
def plot_viscosity(
        data: orm.ArrayData,
        show_plot: bool = False
    ):
    """Plot viscosity data and fit to the Eyring model."""
    print(data)
    shear_rates = data.get_array('shear_rates')
    viscosities = data.get_array('viscosities')


    def eyring_viscosity(gamma_dot, eta_N, sigma_E):
        tau = eta_N / sigma_E
        return (eta_N / (tau * gamma_dot)) * np.log(tau * gamma_dot + np.sqrt((tau * gamma_dot)**2 + 1))

    fit_successful = True
    eta_N, sigma_E = None, None

    eta_N_guess = np.median(viscosities)
    sigma_E_guess = 1e8
    p0 = [eta_N_guess, sigma_E_guess]

    try:
        popt, pcov = curve_fit(
            eyring_viscosity,
            shear_rates,
            viscosities,
            p0=p0,
            maxfev=10000  # Increase max number of function evaluations just in case
        )
        eta_N, sigma_E = popt
        click.echo(f"Fit successful: eta_N = {eta_N:.2f} mPa.s, sigma_E = {sigma_E:.2e} s")
    except Exception as e:
        fit_successful = False
        click.echo(f"Curve fitting failed: {e}")
        sys.exit(1)

    # Create plot
    fig, ax = plt.subplots()
    ax.loglog(shear_rates, viscosities, 'o', label='MD Data')

    if fit_successful:
        shear_rates_fit = np.logspace(np.log10(min(shear_rates)), np.log10(max(shear_rates)), 200)
        ax.loglog(
            shear_rates_fit,
            eyring_viscosity(shear_rates_fit, eta_N, sigma_E),
            '-', label=f'Fit: $\\eta_N$={eta_N:.2f} mPa·s, $\\sigma_E$={sigma_E:.2e} s'
        )

    ax.set_xlabel('Shear Rate (1/s)')
    ax.set_ylabel('Viscosity (mPa·s)')
    ax.set_title('Viscosity vs Shear Rate (log-log)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--')

    # Save plot
    # buf = io.BytesIO()
    # fig.savefig(buf, format='png')
    # buf.seek(0)

    # plot_node = orm.SinglefileData(file=buf)
    # plot_node.store()

    # Also save a local copy to working directory
    plt.tight_layout()
    fig.savefig('viscosity_fit.png')
    click.echo("Plot saved locally as 'viscosity_fit.png'.")

    if show_plot:
        plt.show()

    # # Output
    # if fit_successful:
    #     self.out('fit_parameters', orm.Dict(dict={'eta_mPas': eta_N, 'sigma': sigma_E}).store())
    # else:
    #     self.out('fit_parameters', orm.Dict(dict={'fit_successful': False}).store())

    # self.out('plot', plot_node)
