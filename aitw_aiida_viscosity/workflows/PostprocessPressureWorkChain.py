import io
import re

from aiida import orm
from aiida.engine import WorkChain, calcfunction
from aiida_shell import launch_shell_job  # assuming you use aiida-shell
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


@calcfunction
def extract_deformation_velocities(mdp_files):
    """Extract deformation velocities from MDP files."""
    velocities = []
    for file_path in mdp_files.get_list():
        with open(file_path, 'r') as f:
            lines = f.readlines()
            content = ''.join(lines)
            match = re.search(r'deform\s*=\s*([-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]+)', content)
            if not match:
                raise ValueError(f"No 'deform' line found in {file_path}!")
            numbers = [float(x) for x in match.group(1).split()]
            deform_velocity = numbers[3]  # 4th number (z direction)
            velocities.append(deform_velocity)  # fourth line
    return orm.List(list=velocities)

class PostprocessPressureWorkChain(WorkChain):
    """Post-processing: extract pressures, compute viscosity curves, plot and fit Eyring model."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input_namespace('edr_files', valid_type=orm.SinglefileData, dynamic=True)
        spec.input('grofile', valid_type=orm.SinglefileData, help='Final equilibrated .gro file.')
        spec.input('mdp_files', valid_type=orm.List, help='List of mdp file paths.')

        spec.outline(
            cls.run_gmx_energy_all,
            cls.extract_deformation_velocities,
            cls.compute_averages,
            cls.create_plot_and_fit,
        )

        spec.output('pressure_averages', valid_type=orm.List)
        spec.output('shear_rates', valid_type=orm.List)
        spec.output('viscosities', valid_type=orm.List)
        spec.output('fit_parameters', valid_type=orm.Dict)
        spec.output('plot', valid_type=orm.SinglefileData)

    def run_gmx_energy_all(self):
        """Run `gmx energy` to extract pressure data from each EDR file."""
        self.ctx.pressure_xvg = {}

        sorted_edr_files = sorted(self.inputs.edr_files.items(), key=lambda kv: int(kv[0].split('_')[1]))

        for i, (key, edr_file) in enumerate(sorted_edr_files):
            result, _ = launch_shell_job(
                'gmx_mpi',
                arguments=f'energy -f {{edr}} -o energy_{i}.xvg',
                nodes={
                    'edr': edr_file,
                    'stdin': orm.SinglefileData.from_string('38\n0\n'),  # Select term 38, confirm with 0
                },
                outputs=[f'energy_{i}.xvg'],
                metadata={
                    'computer': orm.load_computer('Tohtori'),
                    'options': {
                        'resources': {'num_machines': 1},
                        'withmpi': False,
                        'redirect_stderr': True,
                        'filename_stdin': 'stdin'
                    }
                }
            )

            self.report(f"GMX energy extraction result for {key}: {result}")

            energy_file = result.get(f'energy_{i}_xvg', None)
            if energy_file is None:
                raise ValueError(f"Energy file 'energy_{i}.xvg' was not generated successfully.")

            self.ctx.pressure_xvg[str(i)] = energy_file

    def extract_deformation_velocities(self):
        """Extract deformation velocities from mdp files."""
        velocities = extract_deformation_velocities(self.inputs.mdp_files)
        self.ctx.deformation_velocities = velocities.get_list()
        self.report(f"Extracted deformation velocities: {self.ctx.deformation_velocities}")

    def compute_averages(self):
        """Compute average pressures, shear rates, and viscosities."""
        # Extract box length (x-dimension)
        with self.inputs.grofile.open() as f:
            lines = f.readlines()
            box_line = lines[-1].strip()
            box_vectors = [float(x) for x in box_line.split()]
            box_length_nm = box_vectors[0]
        self.report(f"Box length extracted: {box_length_nm} nm")

        # Read pressures
        pressures = []
        for i in sorted(self.ctx.pressure_xvg.keys(), key=int):
            node = self.ctx.pressure_xvg[i]
            with node.open() as file_handle:
                data = np.loadtxt(file_handle, comments=['@', '#'])
            avg_pressure = -np.mean(data[:, 1]) # Convert to a positive value
            pressures.append(avg_pressure)

        self.report(f"Average pressures: {pressures}")

        # Unit conversions and calculations
        shear_rates = []
        viscosities = []

        for deform_vel_nm_per_ps, pressure_bar in zip(self.ctx.deformation_velocities, pressures):
            shear_rate = (deform_vel_nm_per_ps * 1000) / (box_length_nm * 1e-9)  # [1/s]
            pressure_Pa = pressure_bar * 1e5  # [Pa]
            viscosity_Pa_s = pressure_Pa / shear_rate  # [Pa.s]
            viscosity_mPa_s = viscosity_Pa_s * 1000  # [mPa.s]

            shear_rates.append(shear_rate)
            viscosities.append(viscosity_mPa_s)

        self.ctx.shear_rates = shear_rates
        self.ctx.viscosities = viscosities
        self.ctx.pressures = pressures

        # Save to outputs
        self.out('pressure_averages', orm.List(list=pressures).store())
        self.out('shear_rates', orm.List(list=shear_rates).store())
        self.out('viscosities', orm.List(list=viscosities).store())

    def create_plot_and_fit(self):
        """Create log-log plot and fit the Eyring equation."""
        shear_rates = np.array(self.ctx.shear_rates)
        viscosities = np.array(self.ctx.viscosities)

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
            self.report(f"Fit successful: eta_N = {eta_N:.2f} mPa.s, sigma_E = {sigma_E:.2e} s")
        except Exception as e:
            fit_successful = False
            self.report(f"Curve fitting failed: {e}")

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
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        plot_node = orm.SinglefileData(file=buf)
        plot_node.store()

        # Also save a local copy to working directory
        plt.tight_layout()
        fig.savefig('viscosity_fit.png')
        self.report("Plot saved locally as 'viscosity_fit.png'.")

        # Output
        if fit_successful:
            self.out('fit_parameters', orm.Dict(dict={'eta_mPas': eta_N, 'sigma': sigma_E}).store())
        else:
            self.out('fit_parameters', orm.Dict(dict={'fit_successful': False}).store())

        self.out('plot', plot_node)
