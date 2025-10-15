import sys

from aiida import orm
from aiida.cmdline.params import arguments, types
import click
import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
except ImportError as e:
    print('Please install the package with [plotting] extras to use this command.')
    sys.exit(1)

from . import cmd_data


@cmd_data.command('plot-viscosity')
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
@click.option(
    '-o', '--output-file',
    type=click.Path(dir_okay=False, writable=True),
    default='viscosity_fit.png',
    show_default=True,
    help='Path to save the output plot image.'
)
def plot_viscosity(
        data: orm.ArrayData,
        output_file: str,
        show_plot: bool = False,
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
        click.echo(f"Fit successful: ")
        for var, value in [('eta_N', eta_N), ('sigma_E', sigma_E)]:
            click.echo(f'{var:>20s} = {value:13.6e}')
    except Exception as e:
        fit_successful = False
        click.echo(f"Curve fitting failed: {e}")
        sys.exit(1)

    if show_plot:
        messages = []
        for backend in ['Qt5Agg', 'TkAgg']:
            try:
                matplotlib.use(backend)
            except Exception as e:
                messages.append(f"Could not use matplotlib backend '{backend}': {e}")
                continue
            else:
                break
        else:
            for msg in messages:
                click.echo(msg)
            click.echo('Could not set a suitable matplotlib backend to show the plot.')
            show_plot = False

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

    plt.tight_layout()
    fig.savefig(output_file)
    click.echo(f"Plot saved locally as '{output_file}'.")

    if show_plot:
        plt.show()
